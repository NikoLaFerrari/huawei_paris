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
from paradise.common.config import Config

from extractor import Extractor
from interpreter import Interpreter
from predictor import Predictor


class Handler:
    """ Handler Class """
    def __init__(self, trace_paths, config_paths, graph_paths, input_dims):
        """
        cfg = config
        trace_path = path to the trace file to-be-analysed
        trc_dims = dimensions used in trace file
        pred_dims = dimensions to be scaled to during regression/claibration
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
        self.input_dims = input_dims
        self.meta = {
            'trace_dims_list': [],
            'model_name_list': [],
            'recompute_enabled': [],
            'raw_configs': [],
            }
        self.formula = {}
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(curr_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.optimized_coeffs = {"mp": 0, "dp": 0, "ep": 0}

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
        extracts the relevant info (dims, model_name, recompute_enables)
        from the raw config.
        """
        pc = raw_cfg.get("parallel_config", {})
        model_cfg = raw_cfg.get("model", {}).get("model_config", {})
        gbs = model_cfg.get("batch_size")
        dp = pc.get("data_parallel", 1)
        mb_num = pc.get("micro_batch_num", 1)
        mb = max(1, gbs // (dp * mb_num))

        model_name = raw_cfg.get("trainer").get("model_name")
        recompute_enabled = raw_cfg.get("recompute_config").get("recompute")

        return (
            {
                "dp": dp,
                "mp": pc.get("model_parallel", 1),
                "pp": pc.get("pipeline_stage", 1),
                "ep": pc.get("expert_parallel", 1),
                "mb": mb,
                "vpp": model_cfg.get("pp_interleave_num", 1),
            },
            model_name,
            recompute_enabled,
        )

    def _get_cache_path(self, model_name, recompute):
        """
        returns correct cache file name.
        """
        safe_name = ""
        for c in model_name:
            if c.isalnum() or c in ('-', '_'):
                safe_name += c
        filename = f"{safe_name}_{recompute}.json"
        return os.path.join(self.cache_dir, filename)

    def write_to_cache(self, model_name, pctg, recompute):
        """
        Writes self.formula to cache file under model_name.
        """
        file_path = self._get_cache_path(model_name, recompute)
        try:
            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(pctg, f, indent=2)
                print(f"[regression] Saved to Cache: {file_path}")
        except (OSError, TypeError, ValueError) as e:
            print(f"[regression] Cache write Error: {e}")

    def calibrate_coefficients(self, actual_dur):
        """
        finds the optimum coeff values for mp, dp, ep
        to be used inside predictor.py.
        """
        print(
            "[interface] Calibrating overlap"
            "coefficients against actual trace durations..."
        )

        all_bucket_data = []
        vpps = []
        temp_pred = Predictor(self.formula, self.meta['trace_dims_list'][0])

        for dims in self.meta['trace_dims_list']:
            _, buckets = temp_pred.predict_with_breakdown(dims)
            all_bucket_data.append(buckets)
            vpps.append(dims.get("vpp", 1))

        initial_guess = [0.95, 0.1, 0.4]
        bounds = [(0, 1), (0, 1), (0, 1)]

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
            print(f"[interface] Optimized overlap"
                  f"coeffs: {self.optimized_coeffs}")
        else:
            print("[interface] Calibration failed "
                  "to converge, using defaults.")

        return self.optimized_coeffs

    def run_nd(self, config, raw_config):
        """
        runs ND internally to obtain the classification inside ND.
        """
        pc = raw_config.get('parallel_config')
        parallel = {
            'pp': pc.get('pipeline_parallel',1),
            'mp': pc.get('model_parallel',1),
            'dp': pc.get('data_parallel',1),
            'ep': pc.get('expert_parallel',1)
        }
        device_count = parallel['pp'] * parallel['mp'] * parallel['dp'] * parallel['ep']
        machine = Har.Machine(device_count, 2)
        recompute = raw_config.get('recompute_config').get('recompute')
        engine = Par.Parallelize(config, machine, None, [], mppb=recompute)
        space = engine.run_generation_to_ordering(None, None)[0]
        print(f"space: {space}")
        dur = space[3]
        total = sum(dur)
        mapping = {
            "COMPUTE": dur[0] + dur[1] + dur[2],
            "DP_COMM": dur[3],
            "MP_COMM": dur[4],
            "EP_COMM": dur[5],
            "CP_COMM": dur[6],
            "PP_COMM": dur[7],
            "BUBBLE": dur[8],
        }

        pctg = {k: (v / total) * 100 for k, v in mapping.items()}
        return pctg, space[3], space[2]

    def normalise_ratios(self, ratio):
        """
        normalises ratios to be in (0,10)
        """
        exp = []
        for k,v in ratio.items():
            exp.append(maths.floor(maths.log10(abs(v))))
        min_exp = min(exp)
        for k in ratio.keys():
            if ratio[k] != 1.0:
                ratio[k] /= 10**(min_exp+1)
        return ratio

    def ratios(self, all_classifications):
        """
        handles ratios required by perf_esimation.
        """
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
            ratio["COMPUTE"] = cla["COMPUTE"]/((nd_total*nd['COMPUTE'])/100)

            for k in comm:
                if k in cla and cla[k] > 0 and nd[k] > 0:
                    lane_vol = (nd[k] / 100) * nd_total
                    ratio[k] = cla[k] / lane_vol
                else:
                    ratio[k] = 1.0

            ratio = self.normalise_ratios(ratio)
            #bt_pctg = {k: (v / bt_total) * 100 for k, v in cla.items()}
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
        1. makes objects of Extractor, Interpreter & Predictor.
        2. retrieves trace data from extractor, sends to interpreter.
        3. retrieves formulae from interpreter, sends to predictor.
        4. retrieves prediction from predictor, converts to percentage (r_out).
        5. runs ND using run_nd(...), obtains ND classification.
        6. compares bench_tools classfn to ND classfn, obtains ratio.
        7. writes ratio to cache file.
        8. returns r_out.
        """
        set_verbose_level(0)
        print(
                f"[regression] Analysing"
                f"{len(self.paths['trace_paths'])} traces..."
        )
        extractor = Extractor(
                self.paths['trace_paths'],
                self.paths['graph_paths'],
                self.paths['config_paths']
        )
        all_samples, all_classifications = extractor.run_extractor()

        interpreter = Interpreter(all_samples)
        self.formula = interpreter.run_interpreter()

        actual_durations = []
        for raw_cd_dict, _ in all_classifications:
            actual_durations.append(sum(raw_cd_dict.values()))

        self.calibrate_coefficients(actual_durations)

        # This loop is required for obtaining the ratios.
        # Replacing base_dims with self.input_dims will lead to prediction.
        for i in range(len(self.meta['trace_dims_list'])):
            base_dims = self.meta['trace_dims_list'][i]  # self.input_dims
            predictor = Predictor(
                    self.formula,
                    base_dims,
                    coeffs=self.optimized_coeffs
            )
            _, pred_buckets = predictor.predict_with_breakdown(base_dims)

            total_work = sum(pred_buckets.values())
            r_out = {
                lane: (val / total_work) * 100 if total_work > 0 else 0
                for lane, val in pred_buckets.items()
            }
            print(f"[interface] r_out_{i + 1}: {json.dumps(r_out, indent=2)}")

            self.ratios(all_classifications)

        return r_out


def objective_fn(x, actual_times, bucket_data, vpps):
    """
    objective function used by scipy.optimize.minimize
    for calibrate_coefficients.
    EXPERIMENTAL.
    """
    coeffs = {"mp": x[0], "dp": x[1], "ep": x[2]}
    e = 0
    eps = 1e-9  # Prevent log10(0)

    for i, actual_time in enumerate(actual_times):
        comp_bubb = (
            bucket_data[i].get("COMPUTE", 0) + bucket_data[i].get("BUBBLE", 0)
        ) / vpps[i]
        blocking = bucket_data[i].get("MP_COMM", 0) * coeffs["mp"]
        async_comm = max(
            bucket_data[i].get("DP_COMM", 0) * coeffs["dp"],
            bucket_data[i].get("EP_COMM", 0) * coeffs["ep"],
        )

        t_pred = comp_bubb + blocking + async_comm

        safe_pred = max(t_pred, eps)
        safe_actual = max(actual_time, eps)

        e += (maths.log10(safe_pred) - maths.log10(safe_actual)) ** 2

    return e
