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

""" Integration between Regression and ND """

import os
import json
import math as maths
import yaml
from scipy.optimize import minimize

from paradise.logger import set_verbose_level
import paradise.parallelize as Par
import paradise.common.hardware as Har
import paradise.dimensions as Dim
from paradise.common.config import Config

from extractor import Extractor
from interpreter import Interpreter
from predictor import Predictor


class Handler:
    """ Handler Class """
    def __init__(self, trace_paths, config_paths, graph_paths, input_config, input_dims):
        self.paths = {
            'trace_paths':  [trace_paths]  if isinstance(trace_paths,  str) else trace_paths,
            'config_paths': [config_paths] if isinstance(config_paths, str) else config_paths,
            'graph_paths':  [graph_paths]  if isinstance(graph_paths,  str) else graph_paths,
        }
        self.input_dims   = input_dims
        self.input_config = input_config
        self.meta = {
            'trace_dims_list':   [],
            'model_name_list':   [],
            'recompute_enabled': [],
            'raw_configs':       [],
        }
        self.formula = {}
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir           = os.path.join(curr_dir, "cache")
        self.extraction_cache_dir = os.path.join(curr_dir, "extraction_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.optimized_coeffs = {"mp": 0.95, "dp": 0.10, "ep": 0.40}

        for path in self.paths['config_paths']:
            with open(path, "r", encoding='utf-8') as f:
                raw = yaml.safe_load(f)
                self.meta['raw_configs'].append(raw)
                dims, model_name, recompute = self.extract_trace_dims(raw)
                self.meta['trace_dims_list'].append(dims)
                self.meta['model_name_list'].append(model_name)
                self.meta['recompute_enabled'].append(recompute)

    # ------------------------------------------------------------------ #
    #  Config helpers                                                      #
    # ------------------------------------------------------------------ #

    def extract_trace_dims(self, raw_cfg):
        pc         = raw_cfg.get("parallel_config", {})
        model_cfg  = raw_cfg.get("model", {}).get("model_config", {})
        runner_cfg = raw_cfg.get('runner_config')
        gbs        = runner_cfg.get("batch_size")
        dp         = pc.get("data_parallel", 1)
        mb_num     = pc.get("micro_batch_num", 1)
        mb         = max(1, gbs // (dp * mb_num))
        model_name        = raw_cfg.get("trainer").get("model_name")
        recompute_enabled = raw_cfg.get("recompute_config").get("recompute")
        return (
            {
                "dp":  dp,
                "mp":  pc.get("model_parallel", 1),
                "pp":  pc.get("pipeline_stage", 1),
                "ep":  pc.get("expert_parallel", 1),
                "mb":  mb,
                "vpp": model_cfg.get("pp_interleave_num", 1),
            },
            model_name,
            recompute_enabled,
        )

    def _get_cache_path(self, model_name, recompute):
        safe_name = "".join(c for c in model_name if c.isalnum() or c in ('-', '_'))
        return os.path.join(self.cache_dir, f"{safe_name}_{recompute}.json")

    def write_to_cache(self, model_name, pctg, recompute):
        file_path = self._get_cache_path(model_name, recompute)
        try:
            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(pctg, f, indent=2)
            print(f"[regression] Saved to Cache: {file_path}")
        except (OSError, TypeError, ValueError) as e:
            print(f"[regression] Cache write Error: {e}")

    # ------------------------------------------------------------------ #
    #  Extraction cache                                                    #
    # ------------------------------------------------------------------ #

    def _cache_path(self, trace_path):
        import hashlib
        key = hashlib.md5(trace_path.encode()).hexdigest()[:12]
        return os.path.join(self.extraction_cache_dir, f"extract_{key}.pkl")

    def _load_or_extract(self):
        """
        Return (all_samples, all_classifications).

        Loads each trace from its .pkl cache when the cache is newer than the
        trace file AND the cache entry contains lane_totals (written after the
        EWA integration). Otherwise re-extracts.
        """
        import pickle

        trace_paths  = self.paths['trace_paths']
        graph_paths  = self.paths['graph_paths']
        config_paths = self.paths['config_paths']

        cached          = [None] * len(trace_paths)
        missing_indices = []

        for i, tp in enumerate(trace_paths):
            cp = self._cache_path(tp)
            if os.path.exists(cp):
                try:
                    trace_mtime = os.path.getmtime(tp)
                    cache_mtime = os.path.getmtime(cp)
                    if cache_mtime >= trace_mtime:
                        with open(cp, "rb") as f:
                            entry = pickle.load(f)
                        # Reject pre-EWA cache entries that lack lane_totals
                        if entry.get("samples", {}).get("lane_totals") is None:
                            print(
                                f"[extractor] Trace {i+1}: cache missing lane_totals "
                                f"(pre-EWA entry), re-extracting."
                            )
                        else:
                            cached[i] = entry
                            print(f"[extractor] Trace {i+1}: loaded from cache ({cp})")
                except Exception as e:
                    print(f"[extractor] Trace {i+1}: cache load failed ({e}), re-extracting.")
            if cached[i] is None:
                missing_indices.append(i)

        if not missing_indices:
            all_samples         = [c["samples"]        for c in cached]
            all_classifications = [c["classification"] for c in cached]
            return all_samples, all_classifications

        # Extract only the missing traces
        extractor = Extractor(
            [trace_paths[i]  for i in missing_indices],
            [graph_paths[i]  for i in missing_indices],
            [config_paths[i] for i in missing_indices],
        )
        new_samples, new_classifications = extractor.run_extractor()

        for j, i in enumerate(missing_indices):
            cp    = self._cache_path(trace_paths[i])
            entry = {"samples": new_samples[j], "classification": new_classifications[j]}
            try:
                with open(cp, "wb") as f:
                    pickle.dump(entry, f)
                print(f"[extractor] Trace {i+1}: saved to cache ({cp})")
            except Exception as e:
                print(f"[extractor] Trace {i+1}: cache save failed ({e})")
            cached[i] = entry

        all_samples         = [c["samples"]        for c in cached]
        all_classifications = [c["classification"] for c in cached]
        return all_samples, all_classifications

    # ------------------------------------------------------------------ #
    #  Calibration                                                         #
    # ------------------------------------------------------------------ #

    def calibrate_coefficients(self, actual_dur):
        """
        Calibrate overlap coefficients using raw Hockney bucket times.
        Fallback path used when EWA lane_totals are unavailable.

        Each trace gets its own Predictor (base_dims == pred_dims) so all
        scale factors reduce to 1.0 — no extrapolation artefacts.
        """
        print("[interface] Calibrating overlap coefficients (Hockney path)...")

        all_bucket_data = []
        vpps = []
        for dims in self.meta['trace_dims_list']:
            per_trace_pred = Predictor(self.formula, dims)
            _, breakdown   = per_trace_pred.predict_with_breakdown(dims)
            all_bucket_data.append(breakdown["bucket_times"])
            vpps.append(dims.get("vpp", dims.get("VPP", 1)))

        res = minimize(
            objective_fn,
            [0.95, 0.1, 0.4],
            args=(actual_dur, all_bucket_data, vpps),
            bounds=[(0, 1)] * 3,
            method="L-BFGS-B",
        )
        if res.success:
            self.optimized_coeffs = {"mp": res.x[0], "dp": res.x[1], "ep": res.x[2]}
            print(f"[interface] Optimized overlap coeffs: {self.optimized_coeffs}")
        else:
            print(f"[interface] Calibration did not converge. Keeping defaults: {self.optimized_coeffs}")
        return self.optimized_coeffs

    def _calibrate_from_ewa(self, all_samples, actual_durations):
        """
        """
        print("[interface] EWA path: verifying self-prediction accuracy...")
        excluded = {"UNKNOWN_COMM", "IDLE"}
        for i, sample in enumerate(all_samples):
            lt      = sample["lane_totals"]
            pred    = sum(v for k, v in lt.items() if k not in excluded)
            actual  = actual_durations[i]
            err_pct = (pred - actual) / actual * 100 if actual > 0 else 0.0
            print(
                f"[interface]   Trace {i+1}: "
                f"sum(lane_totals)={pred/1e6:.4f}s  "
                f"actual={actual/1e6:.4f}s  "
                f"discrepancy={err_pct:+.1f}%"
            )
        print("[interface] EWA path: no overlap calibration needed.")

    # ------------------------------------------------------------------ #
    #  ND interface helpers                                                #
    # ------------------------------------------------------------------ #

    def get_strats_from_nd(self):
        with open(self.input_config, 'r') as f:
            raw = yaml.safe_load(f)
        pc = raw.get('parallel_config')
        parallel = {
            'pp': pc.get('pipeline_parallel', 1),
            'mp': pc.get('model_parallel',    1),
            'dp': pc.get('data_parallel',     1),
            'ep': pc.get('expert_parallel',   1),
        }
        device_count = parallel['pp'] * parallel['mp'] * parallel['dp'] * parallel['ep']
        machine   = Har.Machine(device_count, 2)
        recompute = raw.get('recompute_config').get('recompute')
        engine    = Par.Parallelize(
            self.input_config, machine, self.input_dims['gbs'],
            Dim.get_dims(self.input_dims['dims']), mppb=recompute,
        )
        return [strat[0] for strat in engine.run_generation_to_ordering(None, None)]

    def run_nd(self, config, raw_config):
        pc = raw_config.get('parallel_config')
        parallel = {
            'pp': pc.get('pipeline_parallel', 1),
            'mp': pc.get('model_parallel',    1),
            'dp': pc.get('data_parallel',     1),
            'ep': pc.get('expert_parallel',   1),
        }
        device_count = parallel['pp'] * parallel['mp'] * parallel['dp'] * parallel['ep']
        machine  = Har.Machine(device_count, 2)
        recompute = raw_config.get('recompute_config').get('recompute')
        engine   = Par.Parallelize(config, machine, None, [], mppb=recompute)
        space    = engine.run_generation_to_ordering(None, None)[0]
        dur      = space[3]
        total    = sum(dur)
        mapping  = {
            "COMPUTE": dur[0] + dur[1] + dur[2],
            "DP_COMM": dur[3], "MP_COMM": dur[4],
            "EP_COMM": dur[5], "CP_COMM": dur[6],
            "PP_COMM": dur[7], "BUBBLE":  dur[8],
        }
        pctg = {k: (v / total) * 100 for k, v in mapping.items()}
        return pctg, space[3], space[2]

    def normalise_ratios(self, ratio):
        exp = [maths.floor(maths.log10(abs(v))) for v in ratio.values()]
        min_exp = min(exp)
        for k in ratio:
            if ratio[k] != 1.0:
                ratio[k] /= 10 ** (min_exp + 1)
        return ratio

    def ratios(self, all_classifications):
        for i, (model_name, (cla, pctg)) in enumerate(
            zip(self.meta['model_name_list'], all_classifications)
        ):
            print(f"[interface] calling ND on trace {i + 1}...")
            nd, nd_val, nd_total = self.run_nd(
                Config(self.paths['config_paths'][i]), self.meta['raw_configs'][i]
            )
            comm  = ["DP_COMM", "MP_COMM", "EP_COMM", "CP_COMM", "PP_COMM", "BUBBLE"]
            ratio = {"COMPUTE": cla["COMPUTE"] / ((nd_total * nd['COMPUTE']) / 100)}
            for k in comm:
                if k in cla and cla[k] > 0 and nd[k] > 0:
                    ratio[k] = cla[k] / ((nd[k] / 100) * nd_total)
                else:
                    ratio[k] = 1.0
            ratio = self.normalise_ratios(ratio)
            print(f"nd pctg: {json.dumps(nd, indent=2)}")
            print(f"nd actual: {json.dumps(nd_val, indent=2)}")
            print(f"bench_tools actual: {json.dumps(cla, indent=2)}")
            print(f"bench_tools pctg: {json.dumps(pctg, indent=2)}")
            print("\n ─────────────────────────────────────────────── \n")
            print(f"nd v/s bench_tools: {json.dumps(ratio, indent=2)}")
            self.write_to_cache(model_name, ratio, self.meta['recompute_enabled'][i])

    # ------------------------------------------------------------------ #
    #  Main calibration + prediction                                       #
    # ------------------------------------------------------------------ #

    def run_calibration(self):
        """
        Main entry point.

        EWA path  (preferred): uses lane_totals from EventWaitAnalyzer — causally
                   attributed per-lane µs — and applies _compute_scales physics to
                   scale each lane to the target config. Bypasses Hockney fits.

        Hockney path (fallback): used automatically when lane_totals are absent
                   (e.g. old cache entries). Re-fits α,β per primitive and sums.
        """
        set_verbose_level(0)
        print(f"[regression] Analysing {len(self.paths['trace_paths'])} traces...")

        all_samples, all_classifications = self._load_or_extract()

        use_ewa = all(s.get("lane_totals") is not None for s in all_samples)
        if use_ewa:
            print("[interface] EWA lane totals available — using EWA prediction path.")
        else:
            print("[interface] EWA lane totals missing — falling back to Hockney path.")
            interpreter  = Interpreter(all_samples)
            self.formula = interpreter.run_interpreter()

        # Actual step durations for calibration
        actual_durations = []
        for i, sample in enumerate(all_samples):
            if sample.get('mean_step_us') is not None:
                actual_durations.append(sample['mean_step_us'])
            else:
                raw_cd_dict, _ = all_classifications[i]
                actual_durations.append(sum(raw_cd_dict.values()))
                print(
                    f"[interface] WARNING: trace {i+1} has no RunGraph step times, "
                    f"using classified event sum as fallback."
                )

        if use_ewa:
            self._calibrate_from_ewa(all_samples, actual_durations)
        else:
            self.calibrate_coefficients(actual_durations)

        # Build predictor rooted at the first (base) trace
        base_dims = self.meta['trace_dims_list'][0]
        base_cd   = all_samples[0]["lane_totals"] if use_ewa else None
        predictor = Predictor(
            self.formula if not use_ewa else {},
            base_dims,
            coeffs=self.optimized_coeffs,
        )

        # Predict for each strategy
        #nd_strats = self.get_strats_from_nd()
        nd_strats = self.meta['trace_dims_list']
        keys      = [str(k) for k in nd_strats[0].keys()]
        results   = []

        for strat in nd_strats:
            vals      = [int(v) for v in strat.values()]
            pred_dims = {keys[j]: vals[j] for j in range(len(keys))}

            if use_ewa:
                total_time, lane_times = predictor.predict_from_lane_totals(base_cd, pred_dims)
                raw_total = sum(lane_times.values()) or 1.0
                r_out = {lane: (val / raw_total) * 100 for lane, val in lane_times.items()}
            else:
                total_time, pred_breakdown = predictor.predict_with_breakdown(pred_dims)
                bucket_times = pred_breakdown["bucket_times"]
                raw_total    = sum(bucket_times.values()) or 1.0
                r_out = {lane: (val / raw_total) * 100 for lane, val in bucket_times.items()}

            results.append((pred_dims, total_time, r_out))

        results.sort(key=lambda x: x[1])

        # Build measured_map: normalised dims key → actual step µs
        measured_map = {}
        for i, sample in enumerate(all_samples):
            if sample.get("mean_step_us") is not None:
                norm = {k.lower(): v for k, v in self.meta['trace_dims_list'][i].items()}
                measured_map[tuple(sorted(norm.items()))] = sample['mean_step_us']

        # ── Summary table ─────────────────────────────────────────────
        lane_order = ["COMPUTE", "BUBBLE", "PP_COMM", "MP_COMM",
                      "DP_COMM", "EP_COMM", "OPTIMIZER_SWAP", "IDLE", "UNKNOWN_COMM"]
        col_w = 6

        strat_header = "  ".join(f"{k:>{col_w}}" for k in keys)
        lane_headers = "  ".join(f"{l[:7]:>7}" for l in lane_order)
        print(f"\n[interface] ── ND Strategy Ranking (sorted by predicted step time) ──")
        print(
            f"  {'rank':>4}  {strat_header}  {'total(s)':>{col_w}}  "
            f"{'pred(µs)':>10}  {'meas(µs)':>10}  {'err%':>7}  {lane_headers}"
        )
        print(
            f"  {'─'*4}  {'  '.join(['─'*col_w]*len(keys))}  {'─'*col_w}  "
            f"{'─'*10}  {'─'*10}  {'─'*7}  {'  '.join(['─'*7]*len(lane_order))}"
        )

        for rank, (pred_dims, total_time, r_out) in enumerate(results, 1):
            strat_vals = "  ".join(f"{pred_dims[k]:>{col_w}}" for k in keys)
            lane_vals  = "  ".join(f"{r_out.get(l, 0.0):>7.1f}" for l in lane_order)
            key        = tuple(sorted({k.lower(): v for k, v in pred_dims.items()}.items()))
            measured   = measured_map.get(key)
            meas_str   = f"{measured:>10.0f}"  if measured is not None else f"{'—':>10}"
            err_str    = (
                f"{(total_time - measured) / measured * 100:>+7.1f}%"
                if measured is not None else f"{'':>8}"
            )
            print(
                f"  {rank:>4}  {strat_vals}  {total_time/1e6:>{col_w}.4f}  "
                f"{total_time:>10.0f}  {meas_str}  {err_str}  {lane_vals}"
            )

        print()
        return results[0][2]

    # ------------------------------------------------------------------ #
    #  Leave-one-out cross-validation                                      #
    # ------------------------------------------------------------------ #

    def validate(self):
        """
        Leave-one-out cross-validation across all provided traces.

        For each held-out trace i:
          1. Calibrate on the remaining n-1 traces.
          2. Predict step time for trace i's config.
          3. Compare predicted vs actual (mean_step_us).

        Uses EWA path when lane_totals are present; Hockney fallback otherwise.
        Requires at least 2 traces.

        Returns list[dict] with keys:
            trace_index, actual_µs, predicted_µs, abs_error_µs,
            rel_error_%, lane_times
        """
        n = len(self.paths['trace_paths'])
        if n < 2:
            print("[validate] Need at least 2 traces for LOO-CV. Skipping.")
            return []

        all_samples, all_classifications = self._load_or_extract()
        use_ewa = all(s.get("lane_totals") is not None for s in all_samples)

        actual_durations = []
        for i, sample in enumerate(all_samples):
            if sample.get("mean_step_us") is not None:
                actual_durations.append(sample["mean_step_us"])
            else:
                cd, _ = all_classifications[i]
                actual_durations.append(sum(cd.values()))
                print(f"[validate] WARNING: trace {i+1} missing step times, using event sum.")

        results = []

        for held_out in range(n):
            train_idx = [j for j in range(n) if j != held_out]
            print(
                f"\n[validate] LOO fold {held_out+1}/{n}: "
                f"train on {[j+1 for j in train_idx]}, held-out = trace {held_out+1}"
            )

            train_samples   = [all_samples[j]                       for j in train_idx]
            train_classifs  = [all_classifications[j]                for j in train_idx]
            train_dims_list = [self.meta['trace_dims_list'][j]       for j in train_idx]

            train_actual = []
            for j, s in enumerate(train_samples):
                if s.get("mean_step_us") is not None:
                    train_actual.append(s["mean_step_us"])
                else:
                    cd, _ = train_classifs[j]
                    train_actual.append(sum(cd.values()))

            train_coeffs = {"mp": 0.95, "dp": 0.10, "ep": 0.40}

            if use_ewa:
                train_bucket_data = [s["lane_totals"] for s in train_samples]
            else:
                t_traces  = [self.paths['trace_paths'][j]  for j in train_idx]
                t_graphs  = [self.paths['graph_paths'][j]  for j in train_idx]
                t_configs = [self.paths['config_paths'][j] for j in train_idx]
                t_ext     = Extractor(t_traces, t_graphs, t_configs)
                t_samp, _ = t_ext.run_extractor()
                t_formula = Interpreter(t_samp).run_interpreter()
                train_bucket_data = []
                for dims in train_dims_list:
                    p = Predictor(t_formula, dims)
                    _, bd = p.predict_with_breakdown(dims)
                    train_bucket_data.append(bd["bucket_times"])

            vpps = [d.get("vpp", 1) for d in train_dims_list]
            try:
                res = minimize(
                    objective_fn, [0.95, 0.10, 0.40],
                    args=(train_actual, train_bucket_data, vpps),
                    bounds=[(0, 1)] * 3, method="L-BFGS-B",
                )
                if res.success:
                    train_coeffs = {"mp": res.x[0], "dp": res.x[1], "ep": res.x[2]}
            except Exception as e:
                print(f"[validate] Calibration failed: {e}. Using defaults.")

            held_dims = self.meta['trace_dims_list'][held_out]
            base_dims = train_dims_list[0]

            if use_ewa:
                base_cd   = train_samples[0]["lane_totals"]
                predictor = Predictor({}, base_dims, coeffs=train_coeffs)
                predicted_µs, lane_times = predictor.predict_from_lane_totals(base_cd, held_dims)
            else:
                predictor    = Predictor(t_formula, base_dims, coeffs=train_coeffs)
                predicted_µs, breakdown = predictor.predict_with_breakdown(held_dims)
                lane_times   = breakdown["bucket_times"]

            actual_µs   = actual_durations[held_out]
            abs_err     = abs(predicted_µs - actual_µs)
            rel_err_pct = (abs_err / actual_µs * 100) if actual_µs > 0 else float("inf")

            print(
                f"[validate] Trace {held_out+1}: "
                f"actual={actual_µs/1e6:.4f}s  "
                f"predicted={predicted_µs/1e6:.4f}s  "
                f"error={rel_err_pct:.1f}%"
            )
            results.append({
                "trace_index":  held_out,
                "actual_µs":    actual_µs,
                "predicted_µs": float(predicted_µs),
                "abs_error_µs": abs_err,
                "rel_error_%":  rel_err_pct,
                "lane_times":   lane_times,
            })

        mean_rel = sum(r["rel_error_%"] for r in results) / len(results)
        max_rel  = max(r["rel_error_%"] for r in results)
        print(
            f"\n[validate] LOO-CV summary: "
            f"mean error = {mean_rel:.1f}%,  max = {max_rel:.1f}%"
        )
        return results


# ------------------------------------------------------------------ #
#  Objective function                                                   #
# ------------------------------------------------------------------ #

def objective_fn(x, actual_times, bucket_data, vpps):
    """
    Overlap model:
        T = (COMPUTE + BUBBLE)
          + PP_COMM                        fully serialised (coeff = 1)
          + MP_COMM × coeff_mp             blocking allgather/reducescatter
          + max(DP_COMM × coeff_dp,
                EP_COMM × coeff_ep)        async: only the slower one matters
          + OPTIMIZER_SWAP                 weight-offload stalls compute

    Loss: sum of squared log-ratio errors (scale-invariant).

    Excluded:
        UNKNOWN_COMM  — overlap properties unknown
        IDLE          — constant per-kernel launch overhead, not a free param
    """
    mp_coeff, dp_coeff, ep_coeff = x[0], x[1], x[2]
    e   = 0.0
    eps = 1e-9

    for i, actual_time in enumerate(actual_times):
        bt = bucket_data[i]

        compute_bubble = bt.get("COMPUTE", 0.0) + bt.get("BUBBLE", 0.0)
        blocking       = bt.get("PP_COMM", 0.0) + bt.get("MP_COMM", 0.0) * mp_coeff
        async_comm     = max(
            bt.get("DP_COMM", 0.0) * dp_coeff,
            bt.get("EP_COMM", 0.0) * ep_coeff,
        )
        optimizer_swap = bt.get("OPTIMIZER_SWAP", 0.0)

        t_pred = compute_bubble + blocking + async_comm + optimizer_swap

        e += (maths.log10(max(t_pred, eps)) - maths.log10(max(actual_time, eps))) ** 2

    return e
