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
"""test score functions"""
import numpy as np


def mape(pred, real):
    """mean absolute percentage error"""
    return (
        100
        * 1
        / len(real)
        * sum(abs((r - p) / r) for p, r in zip(pred, real) if p > 0 and r > 0)
    )


def r2(pred, real):
    """coefficient of determination"""
    if len(real) < 2:
        return None
    m = np.mean(real)
    return 1 - sum((r - p) ** 2 for p, r in zip(pred, real)) / sum(
        (r - m) ** 2 for p, r in zip(pred, real) if p > 0 and r > 0
    )
