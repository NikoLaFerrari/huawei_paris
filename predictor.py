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
""" Predicts for given target config/dimensions """

class Predictor:
    def __init__(self, model, base_dims, coeffs={"mp": 0.95, "dp": 0.1, "ep": 0.4}):
        self.model = model
        self.seq_len = base_dims.get("seq_len")
        self.world_size = base_dims.get("world_size")
        self.base_dims = base_dims
        self.coeffs = coeffs


    def predict(self, pred_dims):
        """
        mask function to return just predicted total time.
        """
        total_time, _ = self.predict_with_breakdown(pred_dims)
        return total_time


    def predict_with_breakdown(self, pred_dims):
        """
        Main Function.
        1. uses formula of each bucket to predict the time for each bucket.
        2. sums all the durations to obtain total_time and bucket_times.
        3. returns total_time, bucket_times
        """
        dp, mp, pp, ep, mb = [
            pred_dims.get(k, 1) for k in ["dp", "mp", "pp", "ep", "mb"]
        ]
        vpp = max(1, pred_dims.get("vpp", 1))

        scales = {
            "DP_COMM": dp / self.base_dims["dp"],
            "MP_COMM": mp / self.base_dims["mp"],
            "EP_COMM": (dp * mp) / (self.base_dims["dp"] * self.base_dims["mp"]),
            "BUBBLE": (
                (pp - 1) / max(1, self.base_dims["pp"] - 1)
                if self.base_dims["pp"] > 1
                else 1.0
            ),
            "COMPUTE": mb / self.base_dims["mb"],
            "PP_COMM": 1.0,  # Point-to-Point stays constant per transfer
        }

        primitive_times = {}
        for bucket, params_list in self.model.items():
            f = params_list[0]
            lane = bucket.split("::")[0]
            t_scale = scales.get(lane, 1.0)

            projected_time = (f["alpha"] + (f["beta"] * t_scale)) * f["count_per_step"]
            primitive_times[bucket] = projected_time

        bucket_times = {lane: 0.0 for lane in scales.keys()}
        for key, val in primitive_times.items():
            lane = key.split("::")[0]
            if lane in bucket_times:
                bucket_times[lane] += val

        execution_lane = (
            bucket_times.get("COMPUTE", 0) + bucket_times.get("BUBBLE", 0)
        ) / vpp
        blocking_net = bucket_times.get("MP_COMM", 0) * self.coeffs["mp"]
        async_net = max(
            bucket_times.get("DP_COMM", 0) * self.coeffs["dp"],
            bucket_times.get("EP_COMM", 0) * self.coeffs["ep"],
        )
        total_time = execution_lane + blocking_net + async_net

        return float(total_time), bucket_times
