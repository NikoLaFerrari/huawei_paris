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
"""All Tests"""
import shutil
import os

from perf_estimation.estimate import *
from perf_estimation.utils.gen import generate
from perf_estimation.utils_classes import (
    RatioType,
    PerformanceType,
    P2PCommType,
    RecType,
    CustomConfig,
    PerformanceType,
)
from paradise.common.cost_model_preprocess import *
from paradise.logger import logger

LAST_CONFIGS_PATH = "./toolkit/perf_estimation/last_configs/"
if os.path.exists(LAST_CONFIGS_PATH):
    shutil.rmtree(LAST_CONFIGS_PATH)
os.makedirs(LAST_CONFIGS_PATH)

LAST_SCORES_PATH = LAST_CONFIGS_PATH + "scores/"
if os.path.exists(LAST_SCORES_PATH):
    shutil.rmtree(LAST_SCORES_PATH)
os.makedirs(LAST_SCORES_PATH)

MF_CONFIGS_PATH = "./toolkit/perf_estimation/mf_configs/"
assert os.path.exists(MF_CONFIGS_PATH) and "Bad mindformers configs path"

# modify for more custom configs
rtypes = [rtype for rtype in RatioType]
ttypes = [PerformanceType.FLOP]
ptypes = [ptype for ptype in P2PCommType]
retypes = [retype for retype in RecType]

all_configs = [
    (
        f"{rtype.name[:4]}_{ttype.name[:4]}_{ptype.name[:4]}_{retype.name[:4]}",
        CustomConfig(rtype, ttype, ptype, retype),
    )
    for rtype in rtypes
    for ttype in ttypes
    for ptype in ptypes
    for retype in retypes
]

models = [
    "llama2_7b",
    "llama2_13b",
    "llama2_70b",
    "llama3_70b",
    "pangualpha_13b",
    "deepseek",
]
########
IDX_MODEL = 5
########
PREFIX_MODEL = models[IDX_MODEL]

# if this file is not specified, the yaml generation will
# explore all the parallelism search space for (d, t, p, vp, b, m) with N=32
CONFIG_LIST_FILE_PATH = (
    "./toolkit/perf_estimation/utils/config_list.txt"  # None
)

# generate all possible yamls + dump config list
generate(
    MF_CONFIGS_PATH + PREFIX_MODEL + ".yaml",
    CONFIG_LIST_FILE_PATH,
    fix_global_batch=None,
    nb_devices=1024,
)
CONFIG_LIST_FILE_PATH = LAST_CONFIGS_PATH + "config_list.txt"

config_cases = []
with open(LAST_CONFIGS_PATH + "config_list.txt", "r") as ff:
    for line in ff:
        values = tuple(map(int, line.split()))
        if len(values) == 6:
            config_cases.append(values)

with open(LAST_SCORES_PATH + "all.txt", "w") as a:
    for prefix, ccfg in all_configs:
        with open(LAST_SCORES_PATH + f"{prefix}_scores.txt", "w") as f:
            f.write(f"{PREFIX_MODEL} 64 devices\n")
            for d, t, p, vp, b, m in config_cases:
                logger.info(
                    f"{PREFIX_MODEL} dp={d} mp={t} "
                    f"pp={p} vp={vp} b={b} m={m} with ccfg={prefix}"
                )
                config_path = (
                    LAST_CONFIGS_PATH
                    + f"{PREFIX_MODEL}_{d}_{t}_{p}_{vp}_{b}_{m}.yaml"
                )
                score = estimate_performance(mf_config=config_path, ccfg=ccfg)
                log = (
                    f"{prefix} \t dp={d} \t mp={t} \t pp={p} "
                    "\t vp={vp} \t b={b} \t m={m} \t {score:.2E} "
                    "\t Comm contribution = {ccfg.comm_contribution:.1f}%"
                )
                if ccfg.rtype == RatioType.STATIC:
                    log += (
                        f" \t Static Comm/Comp Ratio = {ccfg.static_ratio:.1E}"
                    )
                elif ccfg.rtype == RatioType.DYNAMIC:
                    log += f" \t Dynamic Comm/Comp Ratio = {ccfg.dynamic_ratio:.1E}"
                f.write(log + "\n")
                a.write(log + "\n")
        a.write("\n")
