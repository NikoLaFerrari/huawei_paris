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
""" Regress extracted data to generate per-bucket primitive formula """

from scipy import stats
import numpy as np


class Interpreter:
    def __init__(self, all_samples):
        self.all_samples = all_samples
        self.formula = {}

    def run_interpreter(self):
        """
        Main Function.
        1. creates x-y dictionaries from Extractor's trace data.
        2. does regression for each primitive
           to obtain formulae (self.formula).
        3. return self.formula.
        """
        print("[interpreter] Interpretting extracted data...")
        bucket_points = {}
        print("\n")
        for sample in self.all_samples:
            for bucket, points in sample["data"].items():
                if bucket not in bucket_points:
                    bucket_points[bucket] = {"x": [], "y": []}
                for p in points:
                    if p["x"] is not None and isinstance(p["x"], (int, float)):
                        bucket_points[bucket]["x"].append(p["x"])
                        bucket_points[bucket]["y"].append(p["y"])

        for bucket, data in bucket_points.items():
            x_vals = np.array(data["x"])  # /(1024**2)
            y_vals = np.array(data["y"])

            if len(np.unique(x_vals)) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x_vals, y_vals
                )
                confidence = r_value**2
            else:
                print(
                    f"[interpreter] {bucket} x_vals are all equal!!"
                )  # defaulting to fake regression: \n{x_vals}\n")
                avg_y = np.mean(y_vals)
                intercept = avg_y * 0.1
                slope = (avg_y * 0.9) / max(1, x_vals[0])
                slope = slope
                confidence = 0.5

            if bucket == "BUBBLE":
                intercept, slope, confidence = np.mean(y_vals), 0, 1
            else:
                intercept, slope = max(0.0, intercept), max(0.0, slope)

            self.formula[bucket] = [
                {
                    "alpha": max(0.0, float(intercept)),
                    "beta": max(0.0, float(slope)),
                    "variable": "bytes",
                    "count_per_step": len(x_vals) / len(self.all_samples),
                    "confidence": float(confidence),
                }
            ]

            # print(f"[interpreter] {bucket}: {self.formula[bucket]}")
        print("\n")

        return self.formula
