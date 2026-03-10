# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Extractor: parse trace files and IR graphs to produce per-primitive
structured data points for Hockney regression, or load pre-extracted
samples directly from cache pkl files.

Data point schema
─────────────────
Every entry in structured_data[lane::primitive] is a dict:

  Communication primitive
  ───────────────────────
  {
    "x":          float,   # message size in bytes (semantically correct per op type)
    "y":          float,   # duration, µs
    "ts":         float,   # absolute timestamp, µs  (for outlier / step detection)
    "count":      int,     # raw element count from trace args
    "dtype":      str,     # e.g. "BFP16", "FP32"
    "alg_type":   str,     # e.g. "MESH-RING-NHR", "NA-NA-RING"
    "group_size": int,     # actual collective group size p (from group_rank_ids)
    "pass_type":  str,     # "forward" | "backward" | "recompute" | "unknown"
    "op_type":    str,     # "AllGather" | "AllReduce" | "ReduceScatter" | ...
    "ir_size":    int,     # raw IR output tensor size (before correction, for debug)
    "is_fused":   bool,    # True if compiler fusion key present (IR size unreliable)
  }

  Compute primitive
  ─────────────────
  {
    "x":          float,   # best available work metric (FLOPs if derivable, else bytes)
    "x_bytes":    float,   # total output volume in bytes (sum of all output tensors)
    "x_flops":    float,   # FLOPs if derivable from input shapes, else None
    "y":          float,   # duration, µs
    "ts":         float,   # absolute timestamp, µs
    "dtype":      str,     # dtype of first output tensor
    "pass_type":  str,     # "forward" | "backward" | "recompute" | "unknown"
    "op_type":    str,     # "MatMul" | "Transpose" | ...
    "is_fused":   bool,    # True if compiler fusion key present
  }

Why message size must be corrected per collective type
──────────────────────────────────────────────────────
In Hockney T(n) = α + n/β, n is the bytes *each rank transfers*.

  AllGather   : each rank sends its local shard.
                n = input_tensor size  (= output / p)
                Using output size inflates n by factor p.

  ReduceScatter: each rank contributes the full tensor, receives a shard.
                n = input_tensor size  (= output × p)
                Using output size deflates n by factor p.

  AllReduce   : in-place; input == output in size.  Either works.

  AlltoAllV   : variable token routing; IR tensor size may not match
                runtime.  Use count × dtype_size from trace args.

  Send/Recv   : point-to-point; use count × dtype_size from trace args
                since IR size is typically 1 (control token).

Why FLOPs are a better x for MatMul-family compute ops
───────────────────────────────────────────────────────
Output tensor size for MatMul[M,K] × [K,N] is M×N bytes, but the
actual work is 2×M×K×N FLOPs — K is invisible in the output.
Two MatMuls with the same output size but different K values do
different amounts of work and will have different durations.
Using FLOPs as x makes the regression physically correct.

Step isolation
──────────────
Warmup steps (typically 1–2 at the start of a trace) have inflated
durations due to CUDA/ACL graph capture, cold HBM caches, and JIT
compilation. Including warmup events in regression biases every α
upward and contaminates β.

run_extractor() calls _get_steady_state_windows() to identify
steady-state step intervals from RunGraph events and then filters
all comm and compute events to those windows only. _get_classification()
accepts the same windows parameter so that classification totals are
consistent with structured_data.

lane_totals
──────────────────────────────────────────────────────────────────────
Each sample carries a "lane_totals" dict produced by
EventWaitAnalyzer.summarize_wait_causes(). These are per-step µs
values attributed causally from the compute stream perspective.
EWA correctly handles the parallel compute/comm streams — rather than
summing raw event durations (which double-counts overlapping streams),
it attributes time to whichever lane the compute stream was blocked on.

Multi-PID deduplication in _get_classification
───────────────────────────────────────────────
MindSpore trace files may contain RunGraph events from multiple
processes under different PIDs (e.g. rank 0 and rank 1 both present).
find_step_events() returns all of them. We deduplicate by discarding
any RunGraph event that overlaps >50% with the preceding kept event —
these represent the same step seen from a second rank.

Dim key normalisation
─────────────────────
Cache pkl files may store dims under framework-native keys (TP, CP, DP,
PP, EP) rather than the mp/dp/pp/ep convention used by _compute_scales.
load_from_pkls() normalises keys on load via _DIMS_KEY_MAP.
"""

import re
import json
import pickle
import yaml

from bench_tools import prof, ms_trace
from bench_tools.utils.ir_utils import (
    get_largest_graph_from_graph_dir,
    get_scope_op_map,
    get_parallel_dimensions,
)
from bench_tools.results.comm_classification import CommunicationClassifier
from bench_tools.event_wait_analysis import EventWaitAnalyzer


# ---------------------------------------------------------------------------
# Collective types that benefit from using INPUT tensor size as n
# ---------------------------------------------------------------------------
_INPUT_SIZE_OPS = {"AllGather", "ReduceScatter", "AllReduce"}

# Collective types where runtime count × dtype_size is more reliable than IR
_TRACE_SIZE_OPS = {"AlltoAllV", "AlltoAllVC", "Send", "Receive", "Broadcast"}

# MatMul-family op types for which we can derive FLOPs from input shapes
_MATMUL_OPS = {"MatMul", "MatMulExt", "BatchMatMul", "Dense"}

# Dim key normalisation: framework-native → internal convention (all lowercase)
# Only MP is used for model/tensor parallelism — TP is not used in this framework.
_DIMS_KEY_MAP = {
    "MP":  "mp",
    "CP":  "cp",
    "DP":  "dp",
    "PP":  "pp",
    "EP":  "ep",
    "VPP": "vpp",
    "MBS": "mb",
    "MB":  "mb",
}


class Extractor:
    """
    Extracts per-primitive (size, duration, metadata) data points from
    MindSpore trace files and their corresponding IR graphs, or loads
    pre-extracted samples from cache pkl files.
    """

    def __init__(self, trace_paths=None, graph_dirs=None, config_paths=None):
        """
        Parameters
        ----------
        trace_paths  : str | list[str] | None   paths to trace_view.json files
        graph_dirs   : str | list[str] | None   paths to IR graph directories
        config_paths : str | list[str] | None   paths to YAML training configs

        All three may be None when using load_from_pkls() only.
        """
        self.trace_paths  = ([trace_paths]  if isinstance(trace_paths,  str) else trace_paths)  or []
        self.graph_dirs   = ([graph_dirs]   if isinstance(graph_dirs,   str) else graph_dirs)   or []
        self.config_paths = ([config_paths] if isinstance(config_paths, str) else config_paths) or []

        self.lane_map = {
            "DP":                "DP_COMM",
            "OtherComm":         "DP_COMM",
            "OP":                "DP_COMM",
            "Optimizer_Internal":"DP_COMM",
            "GlobalNorm":        "DP_COMM",
            "MP":                "MP_COMM",
            "SP":                "MP_COMM",
            "LoadBalance":       "MP_COMM",
            "EP":                "EP_COMM",
            "PP":                "PP_COMM",
            "Dataset":           "BUBBLE",
        }
        self.cd = {}

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run_extractor(self):
        """
        Parse traces and IR graphs to produce samples.

        Returns
        -------
        all_samples : list[dict]
            One entry per trace:
            {
                "data":         {lane::primitive: [data_point, ...]},
                "dims":         {mp, dp, pp, ep, ...},   normalised keys
                "mean_step_us": float | None,
                "lane_totals":  {lane: µs},
            }
        all_classifications : list[tuple]
            One entry per trace: (raw_cd_dict, pctg_dict)
        """
        all_classifications = []
        all_samples         = []
        comm_classifier     = CommunicationClassifier()

        for i in range(len(self.trace_paths)):
            print(f"[extractor] ── Trace {i + 1} / {len(self.trace_paths)} ──")
            process_info = prof.parse_process_info(self.trace_paths[i])

            windows      = self._get_steady_state_windows(process_info)
            mean_step_us = self._get_mean_step_time(process_info)

            print(f"[extractor] Loading IR graph from: {self.graph_dirs[i]}")
            graph_obj       = get_largest_graph_from_graph_dir(
                self.graph_dirs[i], 0, r"trace_code_graph"
            )
            graph_scope_map = get_scope_op_map(graph_obj)

            with open(self.config_paths[i], "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            raw_dims      = get_parallel_dimensions(cfg)
            parallel_dims = self._normalise_dims(raw_dims)

            structured_data = {}

            # ── Communication events ──────────────────────────────────
            comm_pid    = ms_trace.find_communication_pid(process_info)
            comm_events = ms_trace.find_communication_events(process_info, comm_pid)

            n_comm_total = 0
            n_comm_kept  = 0
            for event in comm_events:
                n_comm_total += 1
                if not self._event_in_windows(event, windows):
                    continue
                n_comm_kept += 1
                point = self._extract_comm_point(
                    event, graph_scope_map, parallel_dims, comm_classifier
                )
                if point is None:
                    continue
                bucket_key = point.pop("_key")
                structured_data.setdefault(bucket_key, []).append(point)

            # ── Compute events ────────────────────────────────────────
            compute_pid    = ms_trace.find_compute_pid(process_info)
            tid            = ms_trace.find_kernels_tid(process_info, compute_pid)
            compute_events = prof.get_thread_events(process_info, compute_pid, tid)

            n_comp_total = 0
            n_comp_kept  = 0
            for event in compute_events:
                n_comp_total += 1
                if not self._event_in_windows(event, windows):
                    continue
                n_comp_kept += 1
                point = self._extract_compute_point(event, graph_scope_map)
                if point is None:
                    continue
                bucket_key = point.pop("_key")
                structured_data.setdefault(bucket_key, []).append(point)

            print(
                f"[extractor] Events kept after step isolation: "
                f"comm {n_comm_kept}/{n_comm_total}, "
                f"compute {n_comp_kept}/{n_comp_total}."
            )

            classification = self._get_classification(
                process_info, graph_scope_map, parallel_dims, windows=windows
            )
            cd, pctg = classification

            all_classifications.append(classification)
            all_samples.append({
                "data":         structured_data,
                "dims":         parallel_dims,
                "mean_step_us": mean_step_us,
                "lane_totals":  cd,
            })

        return all_samples, all_classifications

    def load_from_pkls(self, pkl_paths):
        """
        Load pre-extracted samples directly from cache pkl files,
        bypassing trace and IR graph parsing entirely.

        Handles two cache schemas:
          - Flat:    pkl contains the sample dict directly
          - Wrapped: pkl contains {"samples": <sample_dict>, "classification": ...}

        Dim keys are normalised from framework-native names (TP, CP, DP, PP, EP)
        to the internal convention (mp, cp, dp, pp, ep) used by _compute_scales.

        Parameters
        ----------
        pkl_paths : list[str]

        Returns
        -------
        all_samples         : list[dict]
        all_classifications : list[tuple]
        """
        all_samples         = []
        all_classifications = []

        for path in pkl_paths:
            print(f"[extractor] Loading pkl: {path}")
            with open(path, "rb") as f:
                d = pickle.load(f)

            # Unwrap if necessary
            if isinstance(d, dict) and "samples" in d:
                sample = d["samples"]
            else:
                sample = d

            if not isinstance(sample, dict):
                print(f"[extractor] WARNING: unexpected pkl schema in {path}, skipping.")
                continue

            # Normalise dim keys
            raw_dims = sample.get("dims", {})
            sample["dims"] = self._normalise_dims(raw_dims)

            # Validate lane_totals present
            lt = sample.get("lane_totals")
            if not lt:
                print(
                    f"[extractor] WARNING: {path} has no lane_totals — "
                    f"pre-EWA cache entry. Re-extract from trace for EWA path."
                )
                lt = {}

            total = sum(lt.values()) or 1.0
            pctg  = {k: v / total * 100 for k, v in lt.items()}

            print(
                f"[extractor]   dims={sample['dims']}  "
                f"mean_step={sample.get('mean_step_us', 0)/1e6:.4f}s  "
                f"sum(lt)={sum(lt.values())/1e6:.4f}s"
            )

            all_samples.append(sample)
            all_classifications.append((lt, pctg))

        return all_samples, all_classifications

    # ------------------------------------------------------------------ #
    #  Step isolation helpers                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_steady_state_windows(process_info, warmup_steps=2):
        """
        Return a list of (ts_start, ts_end) intervals covering only
        steady-state training steps.

        Returns None when not enough steps are present (keep all events).
        """
        step_events = ms_trace.find_step_events(process_info, "RunGraph")
        if not step_events:
            print(
                "[extractor] WARNING: no RunGraph step events found; "
                "using all events (warmup not excluded)."
            )
            return None

        if len(step_events) <= warmup_steps:
            print(
                f"[extractor] WARNING: only {len(step_events)} step(s) found, "
                f"need > {warmup_steps} to exclude warmup. Using all events."
            )
            return None

        steady  = step_events[warmup_steps:]
        windows = [(e["ts"], e["ts"] + e["dur"]) for e in steady]
        print(
            f"[extractor] Step isolation: skipping {warmup_steps} warmup step(s), "
            f"keeping {len(windows)} steady-state step(s)."
        )
        return windows

    @staticmethod
    def _event_in_windows(event, windows):
        """
        Return True if the event falls inside any steady-state window,
        or if windows is None (no filtering active).
        """
        if windows is None:
            return True
        ts = event.get("ts", 0.0)
        return any(start <= ts <= end for start, end in windows)

    # ------------------------------------------------------------------ #
    #  Communication extraction                                            #
    # ------------------------------------------------------------------ #

    def _extract_comm_point(self, event, graph_scope_map, parallel_dims, comm_classifier):
        """
        Extract a single communication data point.

        Returns a dict with a "_key" field (lane::primitive) plus all
        metadata fields, or None if the event should be skipped.
        """
        args      = event.get("args", {})
        dur       = event.get("dur", 0.0)
        ts        = event.get("ts",  0.0)
        raw_scope = args.get("mindspore_op", "").removeprefix("Kernel::KernelLaunch::")

        op = graph_scope_map.get(raw_scope) or graph_scope_map.get(self._clean(raw_scope))

        if op is not None:
            ir_class = comm_classifier.communication_classification(op, parallel_dims)
        else:
            ir_class = "unknown"

        lane = self.lane_map.get(ir_class)
        if lane is None:
            lane = "UNKNOWN_COMM" if ir_class == "unknown" else "BUBBLE"

        primitive  = self._clean(raw_scope).split("/")[-1] if raw_scope else "UnknownOp"
        bucket_key = f"{lane}::{primitive}"

        trace_count  = args.get("count", 0)
        dtype_str    = str(args.get("data_type", ""))
        alg_type     = args.get("alg_type", "unknown")
        dtype_bytes  = 2 if "16" in dtype_str else 4
        trace_size   = trace_count * dtype_bytes

        op_type    = op.type if op is not None else primitive.split("-")[0]
        group_size = self._get_group_size(op)
        is_fused   = self._is_fused(op)
        pass_type  = self._get_pass_type_from_scope(raw_scope, op)

        ir_out_size = self._get_output_size(op)
        x = self._get_comm_message_size(op_type, op, trace_size, ir_out_size)

        return {
            "_key":       bucket_key,
            "x":          float(x),
            "y":          float(dur),
            "ts":         float(ts),
            "count":      int(trace_count),
            "dtype":      dtype_str,
            "alg_type":   alg_type,
            "group_size": group_size,
            "pass_type":  pass_type,
            "op_type":    op_type,
            "ir_size":    ir_out_size,
            "is_fused":   is_fused,
        }

    def _get_comm_message_size(self, op_type, op, trace_size, ir_out_size):
        """
        Return the physically correct message size n for a collective.

        AllGather / ReduceScatter / AllReduce: use input tensor size from IR.
        AlltoAllV / AlltoAllVC / Send / Receive / Broadcast: use trace_size.
        Fallback: trace_size.
        """
        if op_type in _INPUT_SIZE_OPS:
            ir_in_size = self._get_input_size(op)
            if ir_in_size and ir_in_size > 1:
                return ir_in_size
            if ir_out_size and ir_out_size > 1:
                return ir_out_size
            return trace_size

        if op_type in _TRACE_SIZE_OPS:
            return trace_size if trace_size > 0 else (ir_out_size or 1)

        if ir_out_size and ir_out_size > 1:
            return ir_out_size
        return trace_size if trace_size > 0 else 1

    # ------------------------------------------------------------------ #
    #  Compute extraction                                                  #
    # ------------------------------------------------------------------ #

    def _extract_compute_point(self, event, graph_scope_map):
        """
        Extract a single compute data point.

        Returns a dict with "_key" plus metadata, or None to skip.
        """
        dur  = event.get("dur", 0.0)
        ts   = event.get("ts",  0.0)
        name = event.get("name", "").lower()

        if dur < 0.5:
            return None

        if name in ("event_wait", "event_record"):
            return {
                "_key":      "BUBBLE::Wait",
                "x":         1.0,
                "x_bytes":   1.0,
                "x_flops":   None,
                "y":         float(dur),
                "ts":        float(ts),
                "dtype":     "unknown",
                "pass_type": "unknown",
                "op_type":   "Wait",
                "is_fused":  False,
            }

        raw_scope = (
            event.get("args", {})
            .get("mindspore_op", "")
            .removeprefix("Kernel::KernelLaunch::")
        )
        if not raw_scope:
            return None

        op         = graph_scope_map.get(raw_scope) or graph_scope_map.get(self._clean(raw_scope))
        primitive  = self._clean(raw_scope).split("/")[-1]
        bucket_key = f"COMPUTE::{primitive}"

        op_type   = op.type if op is not None else primitive.split("-")[0]
        is_fused  = self._is_fused(op)
        pass_type = self._get_pass_type_from_scope(raw_scope, op)
        dtype     = self._get_output_dtype(op)

        x_bytes = float(self._get_total_output_size(op) or 1)
        x_flops = self._get_flops(op_type, op)
        x       = float(x_flops) if x_flops is not None else x_bytes

        return {
            "_key":      bucket_key,
            "x":         x,
            "x_bytes":   x_bytes,
            "x_flops":   float(x_flops) if x_flops is not None else None,
            "y":         float(dur),
            "ts":        float(ts),
            "dtype":     dtype,
            "pass_type": pass_type,
            "op_type":   op_type,
            "is_fused":  is_fused,
        }

    # ------------------------------------------------------------------ #
    #  Classification summary (EventWaitAnalyzer-based)                   #
    # ------------------------------------------------------------------ #

    def _get_classification(self, process_info, graph_scope_map, parallel_dims,
                            windows=None):
        """
        Use EventWaitAnalyzer to get accurate per-lane causal breakdown.

        Returns (raw_cd_dict, pctg_dict).

        raw_cd_dict values are per-step µs from summarize_wait_causes(),
        attributed causally from the compute stream perspective. EWA
        correctly handles the parallel compute/comm streams — rather than
        summing event durations (which double-counts overlapping streams),
        it attributes time to whichever lane the compute stream was blocked on.

        Multi-PID deduplication
        ───────────────────────
        Trace files may contain RunGraph events from multiple ranks. We
        deduplicate by discarding any event that overlaps >50% with the
        preceding kept event — these represent the same step from a second
        rank and would cause EWA to divide per-step values by 2×.

        EWA key → lane mapping
        ──────────────────────
        compute        → COMPUTE
        idle           → IDLE
        PP / p2p       → PP_COMM
        MP / SP        → MP_COMM
        DP             → DP_COMM
        EP             → EP_COMM
        LoadBalance    → EP_COMM
        OtherComm/OP/
        GlobalNorm     → DP_COMM
        optimizer_swap/
        swap_out/in    → OPTIMIZER_SWAP
        unattributed/
        collective     → UNKNOWN_COMM
        Dataset        → BUBBLE
        """
        analyzer        = EventWaitAnalyzer()
        comm_classifier = CommunicationClassifier()

        # ── Build scope → lane map for EWA attribution ────────────────
        comm_pid    = ms_trace.find_communication_pid(process_info)
        comm_events = ms_trace.find_communication_events(process_info, comm_pid)
        comm_classification_res = {}
        for event in comm_events:
            args  = event.get("args", {})
            scope = args.get("mindspore_op", "").removeprefix("Kernel::KernelLaunch::")
            op    = graph_scope_map.get(scope) or graph_scope_map.get(self._clean(scope))
            ir_class = (
                comm_classifier.communication_classification(op, parallel_dims)
                if op is not None else "unknown"
            )
            lane = self.lane_map.get(ir_class)
            if lane is None:
                lane = "UNKNOWN_COMM" if ir_class == "unknown" else "BUBBLE"
            comm_classification_res[scope] = lane

        # ── Run EWA pipeline ──────────────────────────────────────────
        compute_events = sorted(
            analyzer.get_compute_stream_events(process_info),
            key=lambda d: d["ts"]
        )
        wait_events     = analyzer.find_wait_events(compute_events)
        delaying_events = analyzer.find_delaying_events(
            process_info, comm_classification_res
        )
        wait_causes = analyzer.find_wait_causes(wait_events, delaying_events)

        # ── Step events: deduplicate overlapping multi-rank markers ───
        # Traces may contain RunGraph events from 2+ ranks covering the
        # same time windows. EWA divides by n_steps — duplicates halve
        # every per-step value. Deduplicate: if two events overlap by
        # >50% of their duration they represent the same step; keep only
        # the first.
        all_step_events = sorted(
            ms_trace.find_step_events(process_info, "RunGraph"),
            key=lambda d: d["ts"]
        )
        deduped = []
        for e in all_step_events:
            if not deduped:
                deduped.append(e)
                continue
            prev     = deduped[-1]
            prev_end = prev["ts"] + prev["dur"]
            overlap  = max(
                0.0,
                min(prev_end, e["ts"] + e["dur"]) - max(prev["ts"], e["ts"])
            )
            if overlap / max(e["dur"], 1.0) > 0.5:
                continue   # same step from another rank — skip
            deduped.append(e)

        if len(deduped) != len(all_step_events):
            print(
                f"[extractor] Deduped {len(all_step_events)} -> {len(deduped)} "
                f"RunGraph events (overlapping multi-rank step markers removed)."
            )
        step_events = deduped

        # ── Restrict to steady-state windows ─────────────────────────
        if windows is not None:
            step_events = [
                e for e in step_events
                if any(start <= e["ts"] <= end for start, end in windows)
            ]

        if not step_events:
            print("[extractor] WARNING: no step events for classification, returning empty.")
            empty = {k: 0.0 for k in [
                "COMPUTE", "BUBBLE", "PP_COMM", "MP_COMM", "DP_COMM",
                "EP_COMM", "UNKNOWN_COMM", "OPTIMIZER_SWAP", "IDLE",
            ]}
            return empty, {k: 0.0 for k in empty}

        print(
            f"[extractor] EWA: {len(step_events)} step(s), "
            f"mean dur={sum(e['dur'] for e in step_events)/len(step_events)/1e6:.4f}s"
        )

        summary = analyzer.summarize_wait_causes(
            compute_events, wait_events, wait_causes, step_events
        )

        ewa_step_time = summary.get("step_time", 0.0)
        print(f"[extractor] EWA step_time={ewa_step_time/1e6:.4f}s")

        # ── EWA key → lane ────────────────────────────────────────────
        ewa_to_lane = {
            "compute":        "COMPUTE",
            "idle":           "IDLE",
            "PP":             "PP_COMM",
            "MP":             "MP_COMM",
            "SP":             "MP_COMM",
            "DP":             "DP_COMM",
            "EP":             "EP_COMM",
            "LoadBalance":    "EP_COMM",
            "OtherComm":      "DP_COMM",
            "OP":             "DP_COMM",
            "GlobalNorm":     "DP_COMM",
            "optimizer_swap": "OPTIMIZER_SWAP",
            "swap_out":       "OPTIMIZER_SWAP",
            "swap_in":        "OPTIMIZER_SWAP",
            "unattributed":   "UNKNOWN_COMM",
            "Dataset":        "BUBBLE",
            "p2p":            "PP_COMM",
            "collective":     "UNKNOWN_COMM",
        }
        for lane in ["DP_COMM", "MP_COMM", "PP_COMM", "EP_COMM", "UNKNOWN_COMM", "BUBBLE"]:
            ewa_to_lane[lane] = lane

        cd = {
            "COMPUTE":        0.0,
            "BUBBLE":         0.0,
            "PP_COMM":        0.0,
            "MP_COMM":        0.0,
            "DP_COMM":        0.0,
            "EP_COMM":        0.0,
            "UNKNOWN_COMM":   0.0,
            "OPTIMIZER_SWAP": 0.0,
            "IDLE":           0.0,
        }

        ignore_keys = {"step_time", "total"}
        for key, val in summary.items():
            if key in ignore_keys:
                continue
            lane = ewa_to_lane.get(key, "UNKNOWN_COMM")
            cd[lane] += val

        # ── Sanity: warn on large unexplained discrepancy ─────────────
        cd_sum = sum(cd.values())
        if ewa_step_time > 0:
            ratio = cd_sum / ewa_step_time
            if not (0.90 <= ratio <= 1.10):
                unmapped = {k: v for k, v in summary.items()
                            if k not in ignore_keys and k not in ewa_to_lane}
                print(
                    f"[extractor] WARNING: sum(cd)={cd_sum/1e6:.4f}s vs "
                    f"EWA step_time={ewa_step_time/1e6:.4f}s "
                    f"(ratio={ratio:.3f}). Unmapped keys: {unmapped}"
                )

        total = sum(cd.values()) or 1.0
        pctg  = {k: (v / total) * 100 for k, v in cd.items()}
        print(f"[extractor] classification:\n{json.dumps(pctg, indent=2)}")
        self.cd = cd
        return self.cd, pctg

    # ------------------------------------------------------------------ #
    #  IR helper methods                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_input_size(op):
        """Return first input tensor size in bytes, or None."""
        if op is None or not op.input_tensors:
            return None
        for t in op.input_tensors:
            if t is None:
                continue
            s = t.get_size()
            if s and s > 1:
                return s
        return None

    @staticmethod
    def _get_output_size(op):
        """Return first output tensor size in bytes, or None."""
        if op is None or not op.output_tensors:
            return None
        s = op.output_tensors[0].get_size()
        return s if (s and s > 1) else None

    @staticmethod
    def _get_total_output_size(op):
        """Return sum of all output tensor sizes in bytes."""
        if op is None or not op.output_tensors:
            return None
        total = 0
        for t in op.output_tensors:
            if t is None:
                continue
            s = t.get_size()
            if s and s > 0:
                total += s
        return total if total > 0 else None

    @staticmethod
    def _get_output_dtype(op):
        """Return dtype string of the first output tensor, or 'unknown'."""
        if op is None or not op.output_tensors or op.output_tensors[0] is None:
            return "unknown"
        return op.output_tensors[0].type or "unknown"

    @staticmethod
    def _get_group_size(op):
        """Return collective group size from group_rank_ids, or -1."""
        if op is None or not op.has_prim_attrs():
            return -1
        prim = op.prim_attrs()
        if prim is None:
            return -1
        ids = prim.get("group_rank_ids")
        if ids is None:
            return -1
        return len(ids) if hasattr(ids, "__len__") else -1

    @staticmethod
    def _is_fused(op):
        """Return True if op has a compiler fusion key."""
        if op is None or not op.has_cnode_prim_attrs():
            return False
        cpa = op.cnode_prim_attrs()
        return cpa is not None and "related_fusion_key" in cpa

    @staticmethod
    def _get_pass_type_from_scope(raw_scope, op=None):
        """
        Determine forward / backward / recompute pass type.
        Priority: op.is_recompute() > op.is_backward() > op.is_forward()
        > scope string heuristics.
        """
        if op is not None:
            if op.is_recompute():
                return "recompute"
            if op.is_backward():
                return "backward"
            if op.is_forward():
                return "forward"

        if "recompute_Default" in raw_scope:
            return "recompute"
        if "Gradients" in raw_scope:
            return "backward"
        if raw_scope:
            return "forward"
        return "unknown"

    @staticmethod
    def _get_flops(op_type, op):
        """
        Derive FLOPs from input tensor shapes for MatMul-family ops.
        Returns int or None.
        """
        if op_type not in _MATMUL_OPS:
            return None
        if op is None or not op.input_tensors or len(op.input_tensors) < 2:
            return None

        t0 = op.input_tensors[0]
        t1 = op.input_tensors[1]
        if t0 is None or t1 is None:
            return None

        s0 = t0.shape
        s1 = t1.shape
        if s0 is None or s1 is None:
            return None
        if -1 in s0 or -1 in s1:
            return None

        try:
            if op_type == "BatchMatMul" and len(s0) == 3 and len(s1) == 3:
                B, M, K = s0
                _, _, N = s1
                return 2 * B * M * K * N

            if len(s0) >= 2 and len(s1) >= 2:
                M     = s0[-2]
                K     = s0[-1]
                N     = s1[-1]
                batch = 1
                for d in s0[:-2]:
                    batch *= d
                return 2 * batch * M * K * N

        except (TypeError, ValueError):
            pass

        return None

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise_dims(dims):
        """
        Normalise parallel dimension keys from framework-native names
        (TP, CP, DP, PP, EP, VPP, MBS) to the internal convention
        (mp, cp, dp, pp, ep, vpp, mb) used by _compute_scales.

        Unknown keys are lowercased and passed through unchanged.
        """
        normalised = {}
        for k, v in dims.items():
            norm_key = _DIMS_KEY_MAP.get(k.upper(), k.lower())
            normalised[norm_key] = v
        return normalised

    @staticmethod
    def _get_mean_step_time(process_info, warmup_steps=2):
        """
        Return mean wall-clock step duration in µs from RunGraph events.
        Falls back to all steps for short traces.
        Returns None only if no RunGraph events exist.
        """
        step_events = ms_trace.find_step_events(process_info, "RunGraph")
        if not step_events:
            return None
        steady = step_events[warmup_steps:] if len(step_events) > warmup_steps else step_events
        return sum(e["dur"] for e in steady) / len(steady)

    @staticmethod
    def _clean(name):
        """Strip trailing -opN suffixes from IR scope strings."""
        if not name:
            return name
        return re.sub(r"-op\d+$", "", name)
