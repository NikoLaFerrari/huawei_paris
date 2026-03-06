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
structured data points for Hockney regression.

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
"""

import re
import json
import yaml

from bench_tools import prof, ms_trace
from bench_tools.ir import graph as G
from bench_tools.utils.ir_utils import (
    get_largest_graph_from_graph_dir,
    get_scope_op_map,
    get_parallel_dimensions,
)
from bench_tools.results.comm_classification import CommunicationClassifier


# ---------------------------------------------------------------------------
# Collective types that benefit from using INPUT tensor size as n
# ---------------------------------------------------------------------------
_INPUT_SIZE_OPS = {"AllGather", "ReduceScatter", "AllReduce"}

# Collective types where runtime count × dtype_size is more reliable than IR
_TRACE_SIZE_OPS = {"AlltoAllV", "AlltoAllVC", "Send", "Receive", "Broadcast"}

# MatMul-family op types for which we can derive FLOPs from input shapes
_MATMUL_OPS = {"MatMul", "MatMulExt", "BatchMatMul", "Dense"}


class Extractor:
    """
    Extracts per-primitive (size, duration, metadata) data points from
    MindSpore trace files and their corresponding IR graphs.
    """

    def __init__(self, trace_paths, graph_dirs, config_paths):
        """
        Parameters
        ----------
        trace_paths  : str | list[str]   paths to trace_view.json files
        graph_dirs   : str | list[str]   paths to IR graph directories
        config_paths : str | list[str]   paths to YAML training configs
        """
        self.trace_paths  = [trace_paths]  if isinstance(trace_paths,  str) else trace_paths
        self.graph_dirs   = [graph_dirs]   if isinstance(graph_dirs,   str) else graph_dirs
        self.config_paths = [config_paths] if isinstance(config_paths, str) else config_paths

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
        Main entry point.

        Returns
        -------
        all_samples : list[dict]
            One entry per trace:
            {
                "data": {lane::primitive: [data_point, ...]},
                "dims": {TP, CP, DP, PP}
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

            print(f"[extractor] Loading IR graph from: {self.graph_dirs[i]}")
            graph_obj       = get_largest_graph_from_graph_dir(
                self.graph_dirs[i], 0, r"trace_code_graph"
            )
            graph_scope_map = get_scope_op_map(graph_obj)

            with open(self.config_paths[i], "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            parallel_dims = get_parallel_dimensions(cfg)

            structured_data = {}

            # ── Communication events ──────────────────────────────────
            comm_pid    = ms_trace.find_communication_pid(process_info)
            comm_events = ms_trace.find_communication_events(process_info, comm_pid)

            for event in comm_events:
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

            for event in compute_events:
                point = self._extract_compute_point(event, graph_scope_map)
                if point is None:
                    continue
                bucket_key = point.pop("_key")
                structured_data.setdefault(bucket_key, []).append(point)

            classification = self._get_classification(
                process_info, graph_scope_map, parallel_dims
            )
            all_classifications.append(classification)
            all_samples.append({"data": structured_data, "dims": parallel_dims})

        return all_samples, all_classifications

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

        # ── IR graph lookup ───────────────────────────────────────────
        op = graph_scope_map.get(raw_scope) or graph_scope_map.get(self._clean(raw_scope))

        # ── Lane classification ───────────────────────────────────────
        if op is not None:
            ir_class = comm_classifier.communication_classification(op, parallel_dims)
        else:
            ir_class = "unknown"

        lane = self.lane_map.get(ir_class)
        if lane is None:
            lane = "UNKNOWN_COMM" if ir_class == "unknown" else "BUBBLE"

        # ── Primitive name ────────────────────────────────────────────
        primitive  = self._clean(raw_scope).split("/")[-1] if raw_scope else "UnknownOp"
        bucket_key = f"{lane}::{primitive}"

        # ── Trace-level metadata ──────────────────────────────────────
        trace_count  = args.get("count", 0)
        dtype_str    = str(args.get("data_type", ""))
        alg_type     = args.get("alg_type", "unknown")
        dtype_bytes  = 2 if "16" in dtype_str else 4
        trace_size   = trace_count * dtype_bytes   # bytes from trace args

        # ── IR-level metadata ─────────────────────────────────────────
        op_type    = op.type if op is not None else primitive.split("-")[0]
        group_size = self._get_group_size(op)
        is_fused   = self._is_fused(op)
        pass_type  = self._get_pass_type_from_scope(raw_scope, op)

        # ── Semantically correct message size ─────────────────────────
        ir_out_size = self._get_output_size(op)   # raw IR output size (debug)
        x = self._get_comm_message_size(
            op_type, op, trace_size, ir_out_size
        )

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

        AllGather / ReduceScatter / AllReduce
            Use input tensor size from IR when available and valid.
            For AllGather: input = shard = output / p  (output is too large).
            For ReduceScatter: input = full tensor = output × p  (output is too small).
            For AllReduce: input == output; either works but input is consistent.

        AlltoAllV / AlltoAllVC
            IR tensor size reflects the *maximum* buffer, not the actual
            runtime token count. Use count × dtype_size from trace args.

        Send / Receive / Broadcast
            These carry a control token (count=1, FP32) paired with actual
            activation tensors. The actual payload size is in trace_size
            when count > 1; when count == 1 it is a handshake message.
            Use trace_size throughout for consistency.

        Fallback
            If IR lookup failed (op is None) or returned an invalid size,
            fall back to trace_size.
        """
        if op_type in _INPUT_SIZE_OPS:
            # Prefer input tensor size from IR
            ir_in_size = self._get_input_size(op)
            if ir_in_size and ir_in_size > 1:
                return ir_in_size
            # IR fallback: if input size unavailable, use output size
            if ir_out_size and ir_out_size > 1:
                return ir_out_size
            return trace_size

        if op_type in _TRACE_SIZE_OPS:
            # Runtime count is more reliable than IR for variable-size ops
            return trace_size if trace_size > 0 else (ir_out_size or 1)

        # Unknown op type: best effort
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

        # ── Sieve out driver noise ────────────────────────────────────
        if dur < 0.5:
            return None

        # ── Pipeline stall markers → BUBBLE lane ─────────────────────
        if name in ("event_wait", "event_record"):
            return {
                "_key":    "BUBBLE::Wait",
                "x":       1.0,
                "x_bytes": 1.0,
                "x_flops": None,
                "y":       float(dur),
                "ts":      float(ts),
                "dtype":   "unknown",
                "pass_type": "unknown",
                "op_type": "Wait",
                "is_fused": False,
            }

        raw_scope = (
            event.get("args", {})
            .get("mindspore_op", "")
            .removeprefix("Kernel::KernelLaunch::")
        )
        if not raw_scope:
            return None

        op        = graph_scope_map.get(raw_scope) or graph_scope_map.get(self._clean(raw_scope))
        primitive = self._clean(raw_scope).split("/")[-1]
        bucket_key = f"COMPUTE::{primitive}"

        # ── Metadata from IR ──────────────────────────────────────────
        op_type   = op.type if op is not None else primitive.split("-")[0]
        is_fused  = self._is_fused(op)
        pass_type = self._get_pass_type_from_scope(raw_scope, op)
        dtype     = self._get_output_dtype(op)

        # ── Work metric: FLOPs preferred, bytes as fallback ───────────
        x_bytes = float(self._get_total_output_size(op) or 1)
        x_flops = self._get_flops(op_type, op)

        # Use FLOPs as primary x when available (physically meaningful)
        x = float(x_flops) if x_flops is not None else x_bytes

        return {
            "_key":    bucket_key,
            "x":       x,
            "x_bytes": x_bytes,
            "x_flops": float(x_flops) if x_flops is not None else None,
            "y":       float(dur),
            "ts":      float(ts),
            "dtype":   dtype,
            "pass_type": pass_type,
            "op_type": op_type,
            "is_fused": is_fused,
        }

    # ------------------------------------------------------------------ #
    #  Classification summary (for calibration / ratio computation)        #
    # ------------------------------------------------------------------ #

    def _get_classification(self, process_info, graph_scope_map, parallel_dims):
        """
        Accumulate total duration per lane for ratio computation.
        Returns (raw_cd_dict, pctg_dict).
        """
        cd = {
            "COMPUTE":      0.0,
            "UNKNOWN_COMM": 0.0,
            "DP_COMM":      0.0,
            "MP_COMM":      0.0,
            "PP_COMM":      0.0,
            "EP_COMM":      0.0,
            "BUBBLE":       0.0,
        }
        comm_classifier = CommunicationClassifier()

        comm_pid    = ms_trace.find_communication_pid(process_info)
        comm_events = ms_trace.find_communication_events(process_info, comm_pid)

        for event in comm_events:
            args      = event.get("args", {})
            dur       = event.get("dur",  0.0)
            raw_scope = args.get("mindspore_op", "").removeprefix("Kernel::KernelLaunch::")
            op        = graph_scope_map.get(raw_scope) or graph_scope_map.get(self._clean(raw_scope))

            ir_class = (
                comm_classifier.communication_classification(op, parallel_dims)
                if op else "unknown"
            )
            lane = self.lane_map.get(ir_class)
            if lane is None:
                lane = "UNKNOWN_COMM" if ir_class == "unknown" else "BUBBLE"

            cd[lane] += dur

        comp_pid    = ms_trace.find_compute_pid(process_info)
        tid         = ms_trace.find_kernels_tid(process_info, comp_pid)
        comp_events = prof.get_thread_events(process_info, comp_pid, tid)

        for event in comp_events:
            dur  = event.get("dur",  0.0)
            name = event.get("name", "").lower()
            if name in ("event_wait", "event_record"):
                cd["BUBBLE"]  += dur
            else:
                cd["COMPUTE"] += dur

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
        """
        Return first input tensor size in bytes, or None if unavailable.
        Used for AllGather / ReduceScatter / AllReduce where input is the
        correct message size for the Hockney model.
        """
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
        """
        Return first output tensor size in bytes, or None.
        """
        if op is None or not op.output_tensors:
            return None
        s = op.output_tensors[0].get_size()
        return s if (s and s > 1) else None

    @staticmethod
    def _get_total_output_size(op):
        """
        Return the sum of all output tensor sizes in bytes.
        Captures ops with multiple outputs (e.g. SplitWithSize).
        """
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
        """
        Return the actual collective group size p from group_rank_ids.
        Falls back to -1 if the attribute is absent (unknown group size).
        """
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
        """
        Return True if the op has a compiler fusion key, meaning its IR
        tensor sizes may not reflect the actual execution boundary.
        """
        if op is None or not op.has_cnode_prim_attrs():
            return False
        cpa = op.cnode_prim_attrs()
        return cpa is not None and "related_fusion_key" in cpa

    @staticmethod
    def _get_pass_type_from_scope(raw_scope, op=None):
        """
        Determine whether this op belongs to the forward pass, backward
        pass, or a recompute section.

        Priority order:
        1. op.is_recompute()  — most reliable (uses duplicated flag + scope)
        2. op.is_backward()   — has forward_unique_id in cnode_primal_attrs
        3. op.is_forward()    — has unique_id in cnode_primal_attrs
        4. Scope string heuristics as fallback when IR lookup failed
        """
        if op is not None:
            if op.is_recompute():
                return "recompute"
            if op.is_backward():
                return "backward"
            if op.is_forward():
                return "forward"

        # Fallback: parse scope string
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

        MatMul / MatMulExt    : inputs [M, K] × [K, N]  →  2·M·K·N
        BatchMatMul           : inputs [B, M, K] × [B, K, N]  →  2·B·M·K·N
        Dense (linear layer)  : inputs [B, in] × [in, out]  →  2·B·in·out

        Returns the FLOP count as an int, or None if shapes are unavailable
        or contain dynamic dimensions (-1).
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
            return None   # Dynamic shape — cannot derive FLOPs statically

        try:
            if op_type == "BatchMatMul" and len(s0) == 3 and len(s1) == 3:
                # [B, M, K] × [B, K, N]
                B, M, K = s0
                _, _, N = s1
                return 2 * B * M * K * N

            if len(s0) >= 2 and len(s1) >= 2:
                # [*, M, K] × [*, K, N]  — handles both 2D and batched
                M = s0[-2]
                K = s0[-1]
                N = s1[-1]
                # Batch dims: product of leading dims of s0
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
    def _clean(name):
        """Strip trailing -opN suffixes from IR scope strings."""
        if not name:
            return name
        return re.sub(r"-op\d+$", "", name)
