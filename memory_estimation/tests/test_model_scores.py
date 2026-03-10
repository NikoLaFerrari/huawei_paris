# Copyright 2025 Huawei Technologies Co., Ltd
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
# pylint: skip-file
import json
from memory_estimation.estimate_v2 import EvaluatorV2
from memory_estimation.score import mape, r2

def test_scores():
    from memory_estimation.hook_base import MemEvalHook

    MemEvalHook.hook_registry.clear()
    from memory_estimation.hooks.texthawk_hooks import XY

    test_peaks, real_peaks = [], []
    changed = False
    # Global MAPE should be <15%
    # Global R2 should be >0.85
    # If model's MAPE is worse, should be <5% difference with current value
    # If model's R2 is worse, should be <0.05 difference with current value
    with open("recorded_scores.json", "r+") as f:
        data = json.load(f)
        n_tests = 0
        logs = ""
        for m in data["models"].keys():
            expected_mem = data["models"][m]["expected_mem"]
            test_m_peaks, real_m_peaks = [], []
            for test in expected_mem:
                if m == "llama3" and test["cfg"] == "vp10_less":
                    test["cfg"] = "vp10.yaml"
                    e = EvaluatorV2(f"../test_cases/{m}/{test['cfg']}", log_level=0)
                    e.set_passes(vpp_less_mem=True)
                elif m == "xy":
                    e = EvaluatorV2(
                        f"../test_cases/{m}/{test['cfg']}", hook_cls=XY(), log_level=0
                    )
                else:
                    e = EvaluatorV2(f"../test_cases/{m}/{test['cfg']}", log_level=0)

                if m == "mixtral":
                    e.set_body_eval_fun(
                        "FULL_REC_LAYER", dyn="dyn_full_rec_layer_gradclip"
                    )

                peak_mem = e.estimate_peak()
                real_m_peaks += [max(test["real"])]
                test_m_peaks += [peak_mem]
            n_tests += len(expected_mem)
            test_m_mape = mape(test_m_peaks, real_m_peaks)
            test_m_r2 = r2(test_m_peaks, real_m_peaks)
            logs += (
                f"Model '{m}'"
                f"\n  MAPE = {test_m_mape}%"
                f"\n  R² = {test_m_r2}"
                f"\n  Num tests = {len(expected_mem)}"
            ) + "\n"
            if test_m_mape >= data["models"][m]["mape"]:
                assert test_m_mape - data["models"][m]["mape"] < 5
            else:
                data["models"][m]["mape"] = test_m_mape
                changed = True
            if test_m_r2 <= data["models"][m]["r2"]:
                assert test_m_r2 - data["models"][m]["r2"] < 0.05
            else:
                data["models"][m]["r2"] = test_m_r2
                changed = True
            real_peaks += real_m_peaks
            test_peaks += test_m_peaks
        test_mape = mape(test_peaks, real_peaks)
        test_r2 = r2(test_peaks, real_peaks)
        print(logs)
        print(
            f"Global MAPE = {test_mape}%"
            f"\nGlobal R² = {test_r2}"
            f"\nTotal num tests = {n_tests}"
        )
        if test_mape >= data["global_mape"]:
            assert test_mape < 15
        else:
            data["global_mape"] = test_mape
            changed = True
        if test_r2 <= data["global_r2"]:
            assert test_r2 > 0.85
        else:
            data["global_r2"] = test_r2
            changed = True
        if changed:
            from datetime import datetime
            from zoneinfo import ZoneInfo

            data["last_modification"] = str(
                datetime.now(ZoneInfo(data["zone"]))
            )
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
