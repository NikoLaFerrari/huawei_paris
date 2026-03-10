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
        """
        cfg = config
        trace_path = path to the trace file to-be-analysed
        trc_dims = dimensions used in trace file
        pred_dims = dimensions to be scaled to during regression/calibration
        formula = holds the formulae
        cache_dir = path to cache/
        """
        self.paths = {
            'trace_paths': (
                [trace_paths] if isinstance(trace_paths, str) else trace_paths
            ),
            'config_paths': (
                 [config_paths] if isinstance(config_paths, str) else config_paths
            ),
            'graph_paths': (
                [graph_paths] if isinstance(graph_paths, str) else graph_paths
            )
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
        self.cache_dir = os.path.join(curr_dir, "cache")
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


    def extract_trace_dims(self, raw_cfg):
        """
        Extracts the relevant info (dims, model_name, recompute_enabled)
        from the raw config.
        """
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
        """Returns correct cache file name."""
        safe_name = ""
        for c in model_name:
            if c.isalnum() or c in ('-', '_'):
                safe_name += c
        filename = f"{safe_name}_{recompute}.json"
        return os.path.join(self.cache_dir, filename)

    def write_to_cache(self, model_name, pctg, recompute):
        """Writes self.formula to cache file under model_name."""
        file_path = self._get_cache_path(model_name, recompute)
        try:
            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(pctg, f, indent=2)
                print(f"[regression] Saved to Cache: {file_path}")
        except (OSError, TypeError, ValueError) as e:
            print(f"[regression] Cache write Error: {e}")

    def calibrate_coefficients(self, actual_dur):
        """
        Find the optimal overlap coefficients (mp, dp, ep) by back-solving:
        "given the raw predicted bucket times, what overlap fractions reproduce
        the observed total step duration?"

        Key correctness requirement
        ───────────────────────────
        For each trace we create a Predictor whose base_dims == pred_dims
        (i.e. the trace's own parallel config). This makes every scale factor
        in _compute_scales reduce to exactly 1.0, so predict_with_breakdown
        simply evaluates the fitted formulae at the training points with no
        extrapolation.

        Using a single shared Predictor with base_dims = trace_dims_list[0]
        for all traces would apply non-unity scales to traces 2, 3, …,
        returning bucket times that reflect a different config rather than
        the one being calibrated against. The optimiser would then fit
        coefficients that compensate for those scale errors, producing
        coefficients that do not generalise to unseen configurations.
        """
        print(
            "[interface] Calibrating overlap coefficients "
            "against actual trace durations..."
        )

        all_bucket_data = []
        vpps = []

        for dims in self.meta['trace_dims_list']:
            # Each trace gets its own Predictor so all scale factors = 1.0.
            per_trace_pred = Predictor(self.formula, dims)
            _, breakdown   = per_trace_pred.predict_with_breakdown(dims)
            all_bucket_data.append(breakdown["bucket_times"])
            vpps.append(dims.get("vpp", dims.get("VPP", 1)))

        initial_guess = [0.95, 0.1, 0.4]
        bounds        = [(0, 1), (0, 1), (0, 1)]

        res = minimize(
            objective_fn,
            initial_guess,
            args=(actual_dur, all_bucket_data, vpps),
            bounds=bounds,
            method="L-BFGS-B",
        )

        if res.success:
            self.optimized_coeffs = {
                "mp": res.x[0],
                "dp": res.x[1],
                "ep": res.x[2],
            }
            print(f"[interface] Optimized overlap coeffs: {self.optimized_coeffs}")
        else:
            print(
                f"[interface] Calibration failed to converge. "
                f"Keeping defaults: {self.optimized_coeffs}"
            )

        return self.optimized_coeffs

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
            self.input_config,
            machine,
            self.input_dims['gbs'],
            Dim.get_dims(self.input_dims['dims']),
            mppb=recompute
        )
        space    = engine.run_generation_to_ordering(None, None)
        nd_dims  = [strat[0] for strat in space]
        return nd_dims

    def run_nd(self, config, raw_config):
        """Runs ND internally to obtain the classification inside ND."""
        pc = raw_config.get('parallel_config')
        parallel = {
            'pp': pc.get('pipeline_parallel', 1),
            'mp': pc.get('model_parallel',    1),
            'dp': pc.get('data_parallel',     1),
            'ep': pc.get('expert_parallel',   1),
        }
        device_count = parallel['pp'] * parallel['mp'] * parallel['dp'] * parallel['ep']
        machine   = Har.Machine(device_count, 2)
        recompute = raw_config.get('recompute_config').get('recompute')
        engine    = Par.Parallelize(config, machine, None, [], mppb=recompute)
        space     = engine.run_generation_to_ordering(None, None)[0]
        dur       = space[3]
        total     = sum(dur)
        mapping   = {
            "COMPUTE": dur[0] + dur[1] + dur[2],
            "DP_COMM": dur[3],
            "MP_COMM": dur[4],
            "EP_COMM": dur[5],
            "CP_COMM": dur[6],
            "PP_COMM": dur[7],
            "BUBBLE":  dur[8],
        }

        pctg = {k: (v / total) * 100 for k, v in mapping.items()}
        return pctg, space[3], space[2]

    def normalise_ratios(self, ratio):
        """Normalises ratios to be in (0, 10)."""
        exp = []
        for k, v in ratio.items():
            exp.append(maths.floor(maths.log10(abs(v))))
        min_exp = min(exp)
        for k in ratio.keys():
            if ratio[k] != 1.0:
                ratio[k] /= 10 ** (min_exp + 1)
        return ratio

    def ratios(self, all_classifications):
        """Handles ratios required by perf_estimation."""
        for i, (model_name, (cla, pctg)) in enumerate(
            zip(self.meta['model_name_list'], all_classifications)
        ):
            print(f"[interface] calling ND on trace {i + 1}...")
            nd, nd_val, nd_total = self.run_nd(
                Config(self.paths['config_paths'][i]), self.meta['raw_configs'][i]
            )
            comm = [
                "DP_COMM",
                "MP_COMM",
                "EP_COMM",
                "CP_COMM",
                "PP_COMM",
                "BUBBLE",
            ]
            ratio = {}
            ratio["COMPUTE"] = cla["COMPUTE"] / ((nd_total * nd['COMPUTE']) / 100)

            for k in comm:
                if k in cla and cla[k] > 0 and nd[k] > 0:
                    lane_vol   = (nd[k] / 100) * nd_total
                    ratio[k]   = cla[k] / lane_vol
                else:
                    ratio[k] = 1.0

            ratio = self.normalise_ratios(ratio)
            print(f"nd pctg: {json.dumps(nd, indent=2)}")
            print(f"nd actual: {json.dumps(nd_val, indent=2)}")
            print(f"bench_tools actual: {json.dumps(cla, indent=2)}")
            print(f"bench_tools pctg: {json.dumps(pctg, indent=2)}")
            print("\n ------------------------------------------------- \n")
            print(f"nd v/s bench_tools :{json.dumps(ratio, indent=2)}")
            self.write_to_cache(
                model_name,
                ratio,
                self.meta['recompute_enabled'][i]
            )

    def run_calibration(self):
        """
        Main Function.
        1. Extracts per-primitive data from trace files and IR graphs.
        2. Fits Hockney formulae via Interpreter.
        3. Back-solves overlap coefficients via calibrate_coefficients.
        4. Predicts for each ND strategy and computes per-lane percentages.
        5. Returns r_out (percentage breakdown) for the last ND strategy.
        """
        set_verbose_level(0)
        print(
            f"[regression] Analysing "
            f"{len(self.paths['trace_paths'])} traces..."
        )

        extractor = Extractor(
            self.paths['trace_paths'],
            self.paths['graph_paths'],
            self.paths['config_paths'],
        )
        all_samples, all_classifications = extractor.run_extractor()

        interpreter  = Interpreter(all_samples)
        self.formula = interpreter.run_interpreter()

        # Total observed step duration per trace (sum of all lane µs).
        actual_durations = [
            sum(raw_cd_dict.values())
            for raw_cd_dict, _ in all_classifications
        ]

        # Calibrate overlap coefficients against the observed durations.
        # Each trace is evaluated with its own Predictor (all scale factors = 1.0)
        # so the optimiser fits true overlap fractions, not scale-error artefacts.
        self.calibrate_coefficients(actual_durations)

        # Build one shared predictor for ND-strategy evaluation.
        # base_dims = first trace config; extrapolation to ND strategies is intended.
        predictor = Predictor(
            self.formula,
            self.meta['trace_dims_list'][0],
            coeffs=self.optimized_coeffs,
        )

        nd_strats = self.get_strats_from_nd()
        keys      = [str(k) for k in nd_strats[0].keys()]
        r_out     = {}

        for i, strat in enumerate(nd_strats):
            vals      = [int(v) for v in strat.values()]
            pred_dims = {keys[j]: vals[j] for j in range(len(keys))}

            _, pred_breakdown = predictor.predict_with_breakdown(pred_dims)
            bucket_times      = pred_breakdown["bucket_times"]

            # Percentages are relative to the sum of raw lane totals, NOT to
            # the overlap-adjusted total_time.  total_time already has comms
            # multiplied by coefficients < 1 and compute divided by vpp, so
            # dividing raw bucket values by it produces numbers that do not
            # sum to 100 % and are misleading.
            raw_total = sum(bucket_times.values())
            r_out = {
                lane: (val / raw_total) * 100 if raw_total > 0 else 0.0
                for lane, val in bucket_times.items()
            }
            print(f"[interface] r_out_{i + 1}: {json.dumps(r_out, indent=2)}")

        return r_out

    def validate(self):
        """
        Leave-one-out cross-validation across the provided traces.

        For each trace i:
          1. Train Extractor + Interpreter on all traces EXCEPT i.
          2. Calibrate overlap coefficients on the training subset.
          3. Predict total step time for trace i's config.
          4. Compare against the actual measured step time for trace i.

        Why this is necessary
        ─────────────────────
        calibrate_coefficients() is in-sample by construction — it
        back-solves from the same traces used for regression. LOO-CV is the
        minimum viable test that confirms the model extrapolates correctly
        to an unseen config. Without it, systematic 2× errors are invisible.

        Requirement: at least 2 trace triplets. With only 1 trace, LOO-CV
        degenerates and a warning is printed instead.

        Returns
        -------
        results : list[dict]
            One entry per trace:
            {
                "trace_index":  int,
                "actual_µs":    float,
                "predicted_µs": float,
                "abs_error_µs": float,
                "rel_error_%":  float,
                "bucket_times": dict,
            }
        """
        n = len(self.paths['trace_paths'])
        if n < 2:
            print("[validate] Need at least 2 traces for LOO-CV. Skipping.")
            return []

        # Obtain actual step durations for all traces using the full extractor.
        full_extractor = Extractor(
            self.paths['trace_paths'],
            self.paths['graph_paths'],
            self.paths['config_paths'],
        )
        _, all_classifications = full_extractor.run_extractor()
        actual_durations = [
            sum(cd.values()) for cd, _ in all_classifications
        ]

        results = []

        for held_out in range(n):
            train_indices = [j for j in range(n) if j != held_out]
            print(
                f"\n[validate] LOO fold {held_out + 1}/{n}: "
                f"training on traces {[j+1 for j in train_indices]}, "
                f"held-out = trace {held_out + 1}"
            )

            # ── Train on all traces except held_out ───────────────────
            train_traces  = [self.paths['trace_paths'][j]  for j in train_indices]
            train_graphs  = [self.paths['graph_paths'][j]  for j in train_indices]
            train_configs = [self.paths['config_paths'][j] for j in train_indices]

            train_extractor              = Extractor(train_traces, train_graphs, train_configs)
            train_samples, train_classifs = train_extractor.run_extractor()

            train_interpreter = Interpreter(train_samples)
            train_formula     = train_interpreter.run_interpreter()

            # Calibrate coefficients on the training subset only.
            train_actual_durs = [sum(cd.values()) for cd, _ in train_classifs]
            train_dims_list   = [self.meta['trace_dims_list'][j] for j in train_indices]

            train_coeffs = {"mp": 0.95, "dp": 0.10, "ep": 0.40}   # fallback defaults
            try:
                all_bucket_data = []
                vpps = []
                for dims in train_dims_list:
                    p = Predictor(train_formula, dims)
                    _, bd = p.predict_with_breakdown(dims)
                    all_bucket_data.append(bd["bucket_times"])
                    vpps.append(dims.get("vpp", dims.get("VPP", 1)))

                res = minimize(
                    objective_fn,
                    [0.95, 0.1, 0.4],
                    args=(train_actual_durs, all_bucket_data, vpps),
                    bounds=[(0, 1)] * 3,
                    method="L-BFGS-B",
                )
                if res.success:
                    train_coeffs = {"mp": res.x[0], "dp": res.x[1], "ep": res.x[2]}
            except Exception as e:
                print(f"[validate] Coefficient calibration failed: {e}. Using defaults.")

            # ── Predict for held-out trace ────────────────────────────
            held_dims = self.meta['trace_dims_list'][held_out]
            base_dims = train_dims_list[0]   # first training trace as base

            predictor    = Predictor(train_formula, base_dims, coeffs=train_coeffs)
            predicted_µs, breakdown = predictor.predict_with_breakdown(held_dims)

            actual_µs   = actual_durations[held_out]
            abs_err     = abs(predicted_µs - actual_µs)
            rel_err_pct = (abs_err / actual_µs * 100) if actual_µs > 0 else float("inf")

            print(
                f"[validate] Trace {held_out + 1}: "
                f"actual={actual_µs:.0f} µs, "
                f"predicted={predicted_µs:.0f} µs, "
                f"error={rel_err_pct:.1f}%"
            )

            results.append({
                "trace_index":  held_out,
                "actual_µs":    actual_µs,
                "predicted_µs": float(predicted_µs),
                "abs_error_µs": abs_err,
                "rel_error_%":  rel_err_pct,
                "bucket_times": breakdown["bucket_times"],
            })

        # ── Summary ───────────────────────────────────────────────────
        mean_rel = sum(r["rel_error_%"] for r in results) / len(results)
        max_rel  = max(r["rel_error_%"] for r in results)
        print(
            f"\n[validate] LOO-CV summary: "
            f"mean relative error = {mean_rel:.1f}%, "
            f"max = {max_rel:.1f}%"
        )
        return results


def objective_fn(x, actual_times, bucket_data, vpps):
    """
    Objective function used by scipy.optimize.minimize
    for calibrate_coefficients.

    bucket_data[i] is the flat {lane: µs} dict (bucket_times sub-dict
    from predict_with_breakdown), NOT the full breakdown dict.

    Loss: sum of squared log-ratio errors across all training traces.
    Log-scale residuals are used so that a 2x error on a 1s step and a
    2x error on a 10s step contribute equally, preventing the longer
    trace from dominating the optimisation.

    UNKNOWN_COMM is intentionally excluded — its overlap properties are
    unknown and it is not included in the predictor total.
    PP_COMM is also excluded — its time is embedded in BUBBLE already.
    """
    coeffs = {"mp": x[0], "dp": x[1], "ep": x[2]}
    e      = 0
    eps    = 1e-9  # prevent log10(0)

    for i, actual_time in enumerate(actual_times):
        bt = bucket_data[i]   # flat {lane: float}

        comp_bubb = (
            bt.get("COMPUTE", 0) + bt.get("BUBBLE", 0)
        ) / max(1, vpps[i])

        blocking = bt.get("MP_COMM", 0) * coeffs["mp"]

        async_comm = max(
            bt.get("DP_COMM", 0) * coeffs["dp"],
            bt.get("EP_COMM", 0) * coeffs["ep"],
        )

        t_pred = comp_bubb + blocking + async_comm

        safe_pred   = max(t_pred,      eps)
        safe_actual = max(actual_time, eps)

        e += (maths.log10(safe_pred) - maths.log10(safe_actual)) ** 2

    return e
