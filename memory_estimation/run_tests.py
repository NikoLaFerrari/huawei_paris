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
"""Tests"""
# from memory_estimation.estimate import Evaluator
from memory_estimation.estimate_v2 import EvaluatorV2
from paradise.common.layer_type import LayerType
from memory_estimation.evaluators.layer_block import EvalFFn
from memory_estimation.evaluators.utils import EvalUtils
from memory_estimation.score import mape as base_mape
from memory_estimation.score import r2 as base_r2

from memory_estimation.logger import logger
from paradise.logger import logger as paradise_logger

import logging
import os
import numpy as np
import json

PLOT = 0  # Plot results
VERBOSE_COMP = 0
LAYER_LOG = 0
SPEC_STAGE_ID = -1
PPB = 0
RANGE_ONLY = 0
UNCERTAINTY = 1024
EVAL = EvaluatorV2
PLAIN = False
# EVAL=Evaluator

current_dir = os.path.dirname(os.path.abspath(__file__))
paradise_logger.setLevel(logging.CRITICAL)


def mape(x, y):
    return round(base_mape(x, y), 2)


def r2(x, y):
    res = base_r2(x, y)
    if not res:
        return None
    return round(res, 2)


def accuracy(x, y):
    if y == 0:
        return y
    if y < 0:
        return "?"
    return int(x / y * 100)


def color(val):
    if not PLAIN:
        res = f"{'+' if val>0 else ''}{val}%"
        if abs(val) <= 15:
            res = f"\033[92m{res}\033[00m"
        elif abs(val) <= 20:
            res = f"\033[93m{res}\033[00m"
        elif abs(val) <= 25:
            res = f"\033[33m{res}\033[00m"
        else:
            res = f"\033[91m{res}\033[00m"
        return res
    return val


def compare_bench(
    real_mem, mem_stages, L, parall, device_mem, verbose=False, range_only=True
):
    global SPEC_STAGE_ID
    if SPEC_STAGE_ID >= len(mem_stages) or SPEC_STAGE_ID < 0:
        SPEC_STAGE_ID = -1
    stage_id = 0
    acc = []
    real_peak_id, pred_peak_id = -1, -1
    real_peak, pred_peak = -1, -1
    for real, pred in zip(real_mem, mem_stages):
        pred_stage_mem = pred["Static"] + pred["Dynamic"]
        real_stage_mem = (
            real
            if isinstance(real, int)
            else int(real["Static"]) + int(real["Dynamic"])
        )
        if real_peak < real_stage_mem:
            real_peak = real_stage_mem
            real_peak_id = stage_id
        if pred_peak < pred_stage_mem:
            pred_peak = pred_stage_mem
            pred_peak_id = stage_id
        total = accuracy(pred_stage_mem, real_stage_mem)
        if isinstance(real, dict):
            if verbose and (SPEC_STAGE_ID == -1 or SPEC_STAGE_ID == stage_id):
                logger.info(f"Stage {stage_id} Accuracies:")
                # logger.info(pred)
                # logger.info(real)
                logger.info(
                    f'\tStatic ({pred["Static"]} vs {real["Static"]}): {accuracy(pred["Static"],int(real["Static"]))}%\t'
                    f' p: {accuracy((pred["ModelParameters"]+pred["ExpModelParameters"]),int(real["ModelParameters"])+int(real["ExpModelParameters"]))}%'
                    f' os: {accuracy(pred["OptimizerStates"],int(real["OptimizerStates"]))}%'
                    f' grad: {accuracy(pred["AccumulatedGradients"],int(real["AccumulatedGradients"]))}%'
                )
                logger.info(
                    f'\tDynamic ({pred["Dynamic"]} vs {real["Dynamic"]}): {accuracy(pred["Dynamic"],int(real["Dynamic"]))}%\t'
                    f' attn: {accuracy((pred["Attn"]+pred["Norm"]/2),int(real["Attn"]))}%'
                    f' ffn: {accuracy((pred["FFn"]+pred["Norm"]/2),int(real["FFn"]))}%'
                    f' comm(dp/tp/cp): {accuracy((pred["DP/OP Comm"]+pred["TP Comm"]+pred["CP Comm"]),int(real["TP/DP/CP AllGather"]))}%'
                    f' comm(ep): {accuracy(pred["EP Comm"],int(real["EP Gather"] if "EP A2A/Gather" not in real else real["EP A2A/Gather"]))}%'
                )
                logger.info(
                    f"\tTotal: ({pred_stage_mem} vs {real_stage_mem}) {total}%"
                )
            acc += [total]
        else:
            if verbose and (SPEC_STAGE_ID == -1 or SPEC_STAGE_ID == stage_id):
                logger.info(
                    f"Stage {stage_id}: ({pred_stage_mem} vs {real_stage_mem}) {total}%"
                )
            acc += [accuracy(pred_stage_mem, real_stage_mem)]
        stage_id += 1
    # logger.info(f'Bounds: [{min(acc)-100}%,{max(acc)-100}%], Pred vs Real : peak id {pred_peak_id} {real_peak_id}, fit mem {device_mem>=pred_peak} {device_mem>=real_peak}')
    u = abs(device_mem - pred_peak) <= UNCERTAINTY
    if range_only:
        return [
            L,
            parall,
            (pred_peak, real_peak),
            f"[{color(min(acc)-100)}, {color(max(acc)-100)}]",
            (pred_peak_id, real_peak_id),
            (device_mem >= pred_peak, device_mem >= real_peak),
            device_mem,
            u,
        ]
    return [
        L,
        parall,
        (pred_peak, real_peak),
        [color(a - 100) if isinstance(a, int) else a for a in acc],
        (pred_peak_id, real_peak_id),
        (device_mem >= pred_peak, device_mem >= real_peak),
        device_mem,
        u,
        [pred["Static"] + pred["Dynamic"] for pred in mem_stages],
        [
            (
                real
                if isinstance(real, int)
                else int(real["Static"]) + int(real["Dynamic"])
            )
            for real in real_mem
        ],
    ]


def print_res(res, name, plot=False):
    if PLAIN:
        header = [
            "Config_Name",
            "Num layers",
            "[DP,TP,PP,EP,CP,VPP,OP,GBS,PP_SCHED]",
            "[Pred/Real]_Peak_Mem",
            "Error_Bounds",
            "[Pred/Real]_Peak_PP_ID",
            "[Pred/Real]_Fit",
            "Max_Device_Mem",
            f"Uncertainty_Fit({UNCERTAINTY})",
        ]
        print("Model_name", name)
        print(";".join(header))
        for r in res:
            # print(r[2][6])
            print(";".join([str(c) for c in r]))
    else:
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter, LogLocator

        header = [
            "Config_Name",
            "Num layers",
            "[DP,TP,PP,EP,CP,VPP,GBS,PP_SCHED]",
            "[Pred/Real]_Peak_Mem",
            "Error_Bounds",
            "[Pred/Real]_Peak_PP_ID",
            "[Pred/Real]_Fit",
            "Max_Device_Mem",
            f"Uncertainty_Fit({UNCERTAINTY})",
        ]
        string_res = [header] + [
            [
                (
                    str(c)
                    if not isinstance(c, list)
                    else f'[{",".join([str(i) for i in c])}]'
                )
                for c in r
            ]
            for r in res
        ]
        max_lengths = [
            max(len(r[i]) for r in string_res)
            for i in range(min(4, len(string_res[0])))
        ]
        for r in string_res:
            r_string = [r[i].rjust(max_lengths[i]) for i in range(4)]
            r_string += r[4:]
            # print("  ".join(r_string))
            if r[-1] != "[]":
                print(f"{{'cfg':'{r[0]}','pred':{r[-2]},'real':{r[-1]}}},")
        df = pd.DataFrame(res)
        # print(
        #     f"MAPE (%) = {mape([y[0] for y in df[3]],[y[1] for y in df[3]])}"
        # )
        # print(f"R² = {r2([y[0] for y in df[3]],[y[1] for y in df[3]])}")
        preds = [y[0] for y in df[3] if y[0] >= 0]
        reals = [y[1] for y in df[3] if y[1] >= 0]
        print(f"{{'mape':{mape(preds,reals)},'r2':{r2(preds,reals)}}}")
        if plot:
            logger.info("Plotting predictions in plots/")
            if not os.path.exists("plots"):
                os.makedirs("plots")
            print(f"Plotting {name} ...")
            x = np.arange(len(df[0]))
            width = 0.35
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(
                x - width / 2,
                [y[1] for y in df[2]],
                width,
                label="Real",
                color="#b3dbfa",
            )
            ax.bar(
                x + width / 2,
                [y[0] for y in df[2]],
                width,
                label="Prediction",
                color="#0087ee",
            )
            for i in range(len(x)):
                p, r = df.loc[i, 2]
                acc = accuracy(p, r)
                if acc != "?":
                    mx = max(p, r)
                    fit_p, fit_r = df.loc[i, 5]
                    ax.text(
                        x[i],
                        mx + 0.025 * mx,
                        f"{'+' if acc-100>0 else ''}{acc-100}%",
                        ha="center",
                        fontsize=7,
                        color="black" if fit_p == fit_r else "red",
                    )
                    if i == 0:
                        ax.hlines(
                            int(df[6][i]),
                            x[i] - width,
                            x[i] + width,
                            color="red",
                            linewidth=0.75,
                            label="Device Memory",
                            ls="dotted",
                        )
                    else:
                        ax.hlines(
                            int(df[6][i]),
                            x[i] - width,
                            x[i] + width,
                            color="red",
                            linewidth=0.75,
                            ls="dotted",
                        )

            ax.set_ylabel("Peak memory (MB)")
            ax.set_xlabel("Configs")
            ax.set_title(name)
            ax.set_xticks(x)
            ax.set_xticklabels(df[0], rotation=45, ha="right", fontsize=7)
            ax.set_yscale("log")
            # plt.axhline(y=device_capacity, xmin=0, xmax=1,color='red',linewidth = 1,label='Device Memory',ls='--')
            ax.yaxis.set_major_locator(
                LogLocator(base=10.0, subs="auto", numticks=10)
            )
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(style="plain", axis="y")
            ax.legend()
            fig.text(
                0.01,
                0.01,
                f"MAPE (%) = {mape([y[0] for y in df[3]],[y[1] for y in df[3]])} \n"
                f"R² = {r2([y[0] for y in df[3]],[y[1] for y in df[3]])}",
                ha="left",
                fontsize=10,
            )
            plt.tight_layout()
            plt.savefig(
                "plots/" + name + "_plot.png", dpi=300, bbox_inches="tight"
            )
            plt.clf()
            plt.close()


def test_estim_functions(
    yml, evaluator, real_mem=None, extra=None, show_res=True, eval_yml=None
):
    res = []
    if isinstance(yml, str):
        if eval_yml:
            logger.info(f"Evaluator uses eval_yml: {eval_yml}")
            e = evaluator(yml, eval_yml=eval_yml)
        else:
            e = evaluator(yml)
        file_name = yml.split(".")[0].split("/")[-1]
    else:
        e = yml
        file_name = e.config_path.split(".")[0].split("/")[-1]
    if LAYER_LOG:
        e.estimate_peak(verbose=True, spec_stage_id=SPEC_STAGE_ID)
        e.reset_config()
    if real_mem:
        e.set_ccfg(extra)
        mem_stages = e.estimate_peak_insight()
        strat = e.get_strategy()
        max_mem = e.get_max_device_memory()
        if "dp" in strat:
            paradise = (
                strat["dp"],
                strat["tp"],
                strat["pp"],
                strat["ep"],
                strat["cp"],
                strat["vpp"],
                strat["op"],
                strat["gbs"],
                strat["sched"],
            )
        else:
            paradise = (
                tuple(strat[m]["dp"] for m in strat.keys()),
                tuple(strat[m]["tp"] for m in strat.keys()),
                tuple(strat[m]["pp"] for m in strat.keys()),
                tuple(strat[m]["ep"] for m in strat.keys()),
                tuple(strat[m]["cp"] for m in strat.keys()),
                tuple(strat[m]["vpp"] for m in strat.keys()),
                tuple(strat[m]["op"] for m in strat.keys()),
                tuple(strat[m]["gbs"] for m in strat.keys()),
                tuple(strat[m]["sched"] for m in strat.keys()),
            )
        c = compare_bench(
            real_mem,
            mem_stages,
            e.get_num_layers(),
            parall,
            max_mem / 1024 / 1024,
            verbose=VERBOSE_COMP,
            range_only=RANGE_ONLY,
        )
        if c[1]:
            res = [[file_name] + c]
        if show_res:
            print_res(res, "Test", plot=False)
    if PPB:
        e.reset_config()
        print(json.dumps(e.estimate_layer_memory(), indent=2))
    return res


def init_real_mem(model_p, exp_model_p, os, g, attn, ffn, ag, a2a, s, d):
    return {
        "ModelParameters": model_p,
        "ExpModelParameters": exp_model_p,
        "OptimizerStates": os,
        "AccumulatedGradients": g,
        "Attn": attn,
        "FFn": ffn,
        "TP/DP/CP AllGather": ag,
        "EP Gather": a2a,
        "Static": s,
        "Dynamic": d,
    }


"""Test Cases"""


class Tests:
    def test_llama3():
        logger.info("TEST LLAMA3")
        files = sorted(
            list(
                set(
                    [
                        n
                        for n in os.listdir(
                            os.path.join(current_dir, "test_cases/llama3/")
                        )
                        if n.endswith("yaml")
                    ]
                )
            )
            + ["vp10_less"]
        )
        res = []
        for f in files:
            yaml_name = f
            if yaml_name == "vp10_less":
                yaml_name = "vp10.yaml"
                e = EVAL(
                    os.path.join(current_dir, "test_cases/llama3/" + yaml_name)
                )
                e.set_passes(vpp_less_mem=True)
            else:
                e = EVAL(
                    os.path.join(current_dir, "test_cases/llama3/" + yaml_name)
                )
            if LAYER_LOG:
                e.estimate_peak(verbose=True, spec_stage_id=SPEC_STAGE_ID)
                e.reset_config()
            mem_stages = e.estimate_peak_insight()
            real_mem = []
            for root, _, files in os.walk(
                os.path.join(
                    current_dir, "test_cases/llama3/" + f.split(".")[0]
                )
            ):
                if len(files) > 1:
                    files = sorted(files, key=lambda x: int(x.split("_")[1]))
                for prof in files:
                    with open(os.path.join(root, prof)) as content:
                        real_mem += [
                            dict(
                                s.strip().split(",")
                                for s in content.readlines()
                            )
                        ]
            strat = e.get_strategy()
            max_mem = e.get_max_device_memory()
            c = compare_bench(
                real_mem,
                mem_stages,
                e.get_num_layers(),
                (
                    strat["dp"],
                    strat["tp"],
                    strat["pp"],
                    strat["ep"],
                    strat["cp"],
                    strat["vpp"],
                    strat["op"],
                    strat["gbs"],
                    strat["sched"],
                ),
                max_mem / 1024 / 1024,
                verbose=VERBOSE_COMP,
                range_only=RANGE_ONLY,
            )
            if c[1]:
                res += [[f] + c]
            if PPB:
                e.reset_config()
                print(json.dumps(e.estimate_layer_memory(), indent=2))
        print_res(res, "LLama3", plot=PLOT)
        return res

    def test_llama2():
        logger.info("TEST LLAMA2")
        files = sorted(
            list(
                set(
                    [
                        n
                        for n in os.listdir(
                            os.path.join(current_dir, "test_cases/llama2/")
                        )
                        if n.endswith("yaml")
                    ]
                )
            )
        )
        res = []
        for f in files:
            e = EVAL(os.path.join(current_dir, "test_cases/llama2/" + f))
            if LAYER_LOG:
                e.estimate_peak(verbose=True, spec_stage_id=SPEC_STAGE_ID)
                e.reset_config()
            mem_stages = e.estimate_peak_insight()
            real_mem = []
            for root, _, files in os.walk(
                os.path.join(
                    current_dir, "test_cases/llama2/" + f.split(".")[0]
                )
            ):
                if len(files) > 1:
                    files = sorted(
                        files, key=lambda x: int(x.split("_")[1][1:])
                    )
                for prof in files:
                    with open(os.path.join(root, prof)) as content:
                        real_mem += [
                            dict(
                                s.strip().split(",")
                                for s in content.readlines()
                            )
                        ]
            strat = e.get_strategy()
            max_mem = e.get_max_device_memory()
            c = compare_bench(
                real_mem,
                mem_stages,
                e.get_num_layers(),
                (
                    strat["dp"],
                    strat["tp"],
                    strat["pp"],
                    strat["ep"],
                    strat["cp"],
                    strat["vpp"],
                    strat["op"],
                    strat["gbs"],
                    strat["sched"],
                ),
                max_mem / 1024 / 1024,
                verbose=VERBOSE_COMP,
                range_only=RANGE_ONLY,
            )
            if c[1]:
                res += [[f] + c]
            if PPB:
                e.reset_config()
                print(json.dumps(e.estimate_layer_memory(), indent=2))
        print_res(res, "LLama2", plot=PLOT)
        return res

    def test_mixtral():
        logger.info("TEST MIXTRAL")
        files = sorted(
            list(
                set(
                    [
                        n
                        for n in os.listdir(
                            os.path.join(current_dir, "test_cases/mixtral/")
                        )
                        if n.endswith("yaml")
                    ]
                )
            )
        )
        res = []
        for f in files:
            e = EVAL(os.path.join(current_dir, "test_cases/mixtral/" + f))
            e.set_body_eval_fun(
                "FULL_REC_LAYER", dyn="dyn_full_rec_layer_gradclip"
            )
            if LAYER_LOG:
                e.estimate_peak(verbose=True, spec_stage_id=SPEC_STAGE_ID)
                e.reset_config()
            mem_stages = e.estimate_peak_insight()
            real_mem = []
            for root, _, files in os.walk(
                os.path.join(
                    current_dir, "test_cases/mixtral/" + f.split(".")[0]
                )
            ):
                if len(files) > 1:
                    files = sorted(files, key=lambda x: int(x.split("_")[1]))
                for prof in files:
                    with open(os.path.join(root, prof)) as content:
                        real_mem += [
                            dict(
                                s.strip().split(",")
                                for s in content.readlines()
                            )
                        ]
            strat = e.get_strategy()
            max_mem = e.get_max_device_memory()
            c = compare_bench(
                real_mem,
                mem_stages,
                e.get_num_layers(),
                (
                    strat["dp"],
                    strat["tp"],
                    strat["pp"],
                    strat["ep"],
                    strat["cp"],
                    strat["vpp"],
                    strat["op"],
                    strat["gbs"],
                    strat["sched"],
                ),
                max_mem / 1024 / 1024,
                verbose=VERBOSE_COMP,
                range_only=RANGE_ONLY,
            )
            if c[1]:
                res += [[f] + c]
            if PPB:
                e.reset_config()
                print(json.dumps(e.estimate_layer_memory(), indent=2))
        print_res(res, "Mixtral", plot=PLOT)
        return res

    def test_qwen():
        logger.info("TEST QWEN")
        res = []
        files = sorted(
            list(
                set(
                    [
                        n
                        for n in os.listdir(
                            os.path.join(current_dir, "test_cases/qwen/")
                        )
                        if n.endswith("yaml")
                    ]
                )
            )
        )
        for f in files:
            e = EVAL(os.path.join(current_dir, "test_cases/qwen/" + f))
            if LAYER_LOG:
                e.estimate_peak(verbose=True, spec_stage_id=SPEC_STAGE_ID)
                e.reset_config()
            mem_stages = e.estimate_peak_insight()
            real_mem = []
            for root, _, files in os.walk(
                os.path.join(current_dir, "test_cases/qwen/" + f.split(".")[0])
            ):
                if len(files) > 1:
                    files = sorted(files, key=lambda x: int(x.split("_")[1]))
                for prof in files:
                    with open(os.path.join(root, prof)) as content:
                        real_mem += [
                            dict(
                                s.strip().split(",")
                                for s in content.readlines()
                            )
                        ]
            strat = e.get_strategy()
            max_mem = e.get_max_device_memory()
            c = compare_bench(
                real_mem,
                mem_stages,
                e.get_num_layers(),
                (
                    strat["dp"],
                    strat["tp"],
                    strat["pp"],
                    strat["ep"],
                    strat["cp"],
                    strat["vpp"],
                    strat["op"],
                    strat["gbs"],
                    strat["sched"],
                ),
                max_mem / 1024 / 1024,
                verbose=VERBOSE_COMP,
                range_only=RANGE_ONLY,
            )
            if c[1]:
                res += [[f] + c]
            if PPB:
                e.reset_config()
                print(json.dumps(e.estimate_layer_memory(), indent=2))
        print_res(res, "Qwen", plot=PLOT)
        return res

    def test_deepseek():
        logger.info("TEST DS3")
        res = []
        files = sorted(
            list(
                set(
                    [
                        n
                        for n in os.listdir(
                            os.path.join(current_dir, "test_cases/deepseek3/")
                        )
                        if n.endswith("yaml")
                    ]
                )
            )
        )
        for f in files:
            e = EVAL(os.path.join(current_dir, "test_cases/deepseek3/" + f))
            if LAYER_LOG:
                e.estimate_peak(verbose=True, spec_stage_id=SPEC_STAGE_ID)
                e.reset_config()
            mem_stages = e.estimate_peak_insight()
            real_mem = []
            for root, _, files in os.walk(
                os.path.join(
                    current_dir, "test_cases/deepseek3/" + f.split(".")[0]
                )
            ):
                if len(files) > 1:
                    files = sorted(
                        files, key=lambda x: int(x.split("_")[1][1:])
                    )
                for prof in files:
                    with open(os.path.join(root, prof)) as content:
                        real_mem += [
                            dict(
                                s.strip().split(",")
                                for s in content.readlines()
                            )
                        ]
            strat = e.get_strategy()
            max_mem = e.get_max_device_memory()
            c = compare_bench(
                real_mem,
                mem_stages,
                e.get_num_layers(),
                (
                    strat["dp"],
                    strat["tp"],
                    strat["pp"],
                    strat["ep"],
                    strat["cp"],
                    strat["vpp"],
                    strat["op"],
                    strat["gbs"],
                    strat["sched"],
                ),
                max_mem / 1024 / 1024,
                verbose=VERBOSE_COMP,
                range_only=RANGE_ONLY,
            )
            if c[1]:
                res += [[f.split(".")[0]] + c]
            if PPB:
                e.reset_config()
                print(json.dumps(e.estimate_layer_memory(), indent=2))
        files = sorted(
            list(
                set(
                    [
                        n
                        for n in os.listdir(
                            os.path.join(
                                current_dir, "test_cases/deepseek3/small/"
                            )
                        )
                        if n.endswith("yaml")
                    ]
                )
            )
        )
        for f in files:
            e = EVAL(
                os.path.join(current_dir, "test_cases/deepseek3/small/" + f)
            )
            if LAYER_LOG:
                e.estimate_peak(verbose=True, spec_stage_id=SPEC_STAGE_ID)
                e.reset_config()
            mem_stages = e.estimate_peak_insight()
            real_mem = []
            for root, _, files in os.walk(
                "test_cases/deepseek3/small/" + f.split(".")[0]
            ):
                if len(files) > 1:
                    files = sorted(
                        files, key=lambda x: int(x.split("_")[1][1:])
                    )
                for prof in files:
                    with open(os.path.join(root, prof)) as content:
                        real_mem += [
                            dict(
                                s.strip().split(",")
                                for s in content.readlines()
                            )
                        ]
            strat = e.get_strategy()
            max_mem = e.get_max_device_memory()
            c = compare_bench(
                real_mem,
                mem_stages,
                e.get_num_layers(),
                (
                    strat["dp"],
                    strat["tp"],
                    strat["pp"],
                    strat["ep"],
                    strat["cp"],
                    strat["vpp"],
                    strat["op"],
                    strat["gbs"],
                    strat["sched"],
                ),
                max_mem / 1024 / 1024,
                verbose=VERBOSE_COMP,
                range_only=RANGE_ONLY,
            )
            if c[1]:
                res += [["small_" + f.split(".")[0]] + c]
            if PPB:
                e.reset_config()
                print(json.dumps(e.estimate_layer_memory(), indent=2))
        print_res(res, "DeepSeek3", plot=PLOT)
        return res

    def test_cm():
        logger.info("TEST CM")
        res = []
        path = "/mnt/nvme_1_3_4/philippe/mindformers/configs/llama2/"
        files = sorted(
            list(
                set(
                    [
                        n
                        for n in os.listdir(os.path.join(current_dir, path))
                        if n.endswith("yaml") and n.startswith("cm_")
                    ]
                )
            )
        )
        # files = [files[0]]
        import summarize

        for f in files:
            e = EVAL(os.path.join(current_dir, path + f))
            if LAYER_LOG:
                e.estimate_peak(verbose=True, spec_stage_id=SPEC_STAGE_ID)
                e.reset_config()
            mem_stages = e.estimate_peak_insight()
            # print("")
            real_mem = []
            folder = os.path.join(current_dir, path + f.split(".")[0])
            for root, _, files in os.walk(folder):
                if len(files) > 1:
                    files = sorted(
                        files, key=lambda x: int(x.split("_")[1][1:])
                    )
                for prof in files:
                    real_mem += [
                        summarize.summarize_insight(os.path.join(folder, prof))
                    ]
            strat = e.get_strategy()
            max_mem = e.get_max_device_memory()
            c = compare_bench(
                real_mem,
                mem_stages,
                e.get_num_layers(),
                (
                    strat["dp"],
                    strat["tp"],
                    strat["pp"],
                    strat["ep"],
                    strat["cp"],
                    strat["vpp"],
                    strat["op"],
                    strat["gbs"],
                    strat["sched"],
                ),
                max_mem / 1024 / 1024,
                verbose=VERBOSE_COMP,
                range_only=RANGE_ONLY,
            )
            if c[1]:
                res += [[f.split(".")[0]] + c]
            if PPB:
                e.reset_config()
                print(json.dumps(e.estimate_layer_memory(), indent=2))
        print_res(res, "CM", plot=PLOT)
        return res

    def test_deepseek_gmm():  # Usable in 238
        # HQ Baseline
        # GMM
        real_mem = [
            init_real_mem(
                294, 2698, 5986, 5565, 5196, 3465, 3135, 1792, 14840, 21028
            ),
            init_real_mem(
                136, 4048, 8370, 5822, 3600, 6078, 1473, 2912, 18674, 22133
            ),
            init_real_mem(
                89, 5397, 10974, 6963, 2800, 6511, 1031, 3136, 23723, 21433
            ),
            init_real_mem(
                322, 5397, 11441, 9122, 2000, 4779, 2931, 2240, 26698, 24841
            ),
        ]

        def custom(c):
            c.shard_embed = (
                c.d if (c.vocab_emb_dp or (c.v % c.t != 0)) else (c.t * c.d)
            )

        test_estim_functions(
            "/mnt/nvme_1_3_4/philippe/parall/toolkit/memory_estimation/test_cases/customizable2.yaml",
            EVAL,
            real_mem,
            extra=custom,
        )

        # BMM
        test_estim_functions(
            "/mnt/nvme_1_3_4/philippe/parall/toolkit/memory_estimation/test_cases/customizable.yaml",
            EVAL,
            extra=custom,
        )

        # Xunyi (GMM)
        # test_estim_functions('/mnt/nvme_1_3_4/philippe/config_3f13d8adbfe3.yaml',EVAL, extra=custom)

        # Zijing
        # GMM
        real_mem = [
            init_real_mem(
                2214, 3465, 11359, 13352, 1645, 2904, 9064, 280, 30435, 24621
            )
        ]
        test_estim_functions(
            "/mnt/nvme_1_3_4/philippe/to_philippe/bench_299f64704d4d-gmm/config_299f64704d4d.yaml",
            EVAL,
            real_mem,
            extra=custom,
        )

        # BMM
        real_mem = [
            init_real_mem(
                2214, 3465, 11359, 13352, 1575, 3481, 9064, 280, 30436, 24392
            )
        ]
        test_estim_functions(
            "/mnt/nvme_1_3_4/philippe/to_philippe/bench_c19bf4d382c0-bmm/config_c19bf4d382c0.yaml",
            EVAL,
            real_mem,
            extra=custom,
        )

        # test_estim_functions("/mnt/nvme_1_3_4/philippe/mindformers/research/deepseek3/deepseek3_671b/tinyseek3_8p.yaml",EVAL, overwrite_ccfg_fun=custom)
        return None

    def test_deepseek_dualpipe_512():
        # 512 Devices
        res = []
        real_mem = [
            init_real_mem(
                189, 4723, 9825, 24478, 13086, 17982, 15584, 112, 40436, 63314
            ),
            init_real_mem(
                142, 5229, 10743, 22717, 12601, 22037, 12447, 112, 40073, 55884
            ),
        ]
        r = test_estim_functions(
            "/mnt/nvme_1_3_4/philippe/mindformers/configs/llama2/hq_ds3_dualpipe_nd1_norec.yaml",
            EVAL,
            real_mem,
            show_res=False,
        )
        res += r

        real_mem = [
            init_real_mem(
                189, 4723, 9825, 24478, 261, 5, 14040, 112, 42403, 25768
            ),
            init_real_mem(
                142, 5229, 10743, 22717, 263, 2, 12981, 112, 42035, 24926
            ),
        ]
        r = test_estim_functions(
            "/mnt/nvme_1_3_4/philippe/mindformers/configs/llama2/hq_ds3_dualpipe_nd1_fullrec.yaml",
            EVAL,
            real_mem,
            show_res=False,
        )
        res += r
        real_mem = [
            init_real_mem(
                189, 4723, 9825, 20485, 6576, 9047, 13146, 56, 36217, 37260
            ),
            init_real_mem(
                142, 5229, 10743, 20175, 6332, 11018, 11176, 56, 37307, 32914
            ),
        ]
        r = test_estim_functions(
            "/mnt/nvme_1_3_4/philippe/mindformers/configs/llama2/hq_ds3_dualpipe_nd2_norec.yaml",
            EVAL,
            real_mem,
            show_res=False,
        )
        res += r

        real_mem = [
            init_real_mem(
                189, 4723, 9825, 20485, 131, 2, 11602, 56, 37398, 17855
            ),
            init_real_mem(
                142, 5229, 10743, 20175, 132, 1, 11716, 56, 38480, 17760
            ),
        ]
        r = test_estim_functions(
            "/mnt/nvme_1_3_4/philippe/mindformers/configs/llama2/hq_ds3_dualpipe_nd2_fullrec.yaml",
            EVAL,
            real_mem,
            show_res=False,
        )
        res += r

        real_mem = [
            init_real_mem(
                347, 2698, 6093, 16942, 14937, 2850, 12685, 224, 26410, 51690
            ),
            init_real_mem(
                114, 2698, 5626, 13015, 13601, 4138, 8794, 224, 21799, 42348
            ),
            init_real_mem(
                136, 4048, 8370, 18952, 25602, 24048, 12190, 224, 31902, 85653
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 25602, 42674, 12967, 224, 40611, 100074
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 25602, 42674, 12967, 224, 40611, 100074
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 25602, 42674, 12967, 224, 40611, 100074
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 25602, 42674, 12967, 224, 40611, 100074
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 25602, 42674, 12967, 224, 40611, 100074
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 25602, 42674, 12967, 224, 40610, 100074
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 25602, 42674, 12967, 224, 40610, 100074
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 25602, 42674, 12967, 224, 40610, 100074
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 25602, 42674, 12967, 224, 40610, 100074
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 25602, 42674, 12967, 224, 40610, 100074
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 25602, 42674, 12967, 224, 40610, 100074
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 25602, 42674, 12967, 224, 40610, 100074
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 25601, 41580, 12911, 224, 40610, 97924
            ),
        ]
        r = test_estim_functions(
            "/mnt/nvme_1_3_4/philippe/mindformers/configs/llama2/hq_ds3_dualpipe_nd3_norec.yaml",
            EVAL,
            real_mem,
            show_res=False,
        )
        res += r

        real_mem = [
            init_real_mem(
                347, 2698, 6093, 16942, 520, 0, 10421, 224, 26745, 25072
            ),
            init_real_mem(
                114, 2698, 5626, 13015, 520, 0, 6570, 224, 22133, 14023
            ),
            init_real_mem(
                136, 4048, 8370, 18952, 520, 6, 9542, 224, 32355, 19761
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 521, 5, 11352, 224, 41064, 22432
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 522, 5, 11464, 224, 41064, 23832
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 522, 5, 11576, 224, 41064, 25233
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 523, 5, 11688, 224, 41064, 26633
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 523, 5, 11800, 224, 41064, 28033
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 524, 5, 11912, 224, 41064, 29434
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 524, 5, 12024, 224, 41064, 30834
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 525, 5, 12136, 224, 41064, 32235
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 525, 5, 12248, 224, 41064, 33635
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 526, 5, 12360, 224, 41064, 35036
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 526, 5, 12472, 224, 41064, 36436
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 527, 5, 12584, 224, 41064, 37837
            ),
            init_real_mem(
                89, 5397, 10974, 23747, 527, 5, 12652, 224, 41063, 38591
            ),
        ]
        r = test_estim_functions(
            "/mnt/nvme_1_3_4/philippe/mindformers/configs/llama2/hq_ds3_dualpipe_nd3_fullrec.yaml",
            EVAL,
            real_mem,
            show_res=False,
        )
        res += r

        real_mem = [
            init_real_mem(
                464, 1349, 3628, 9731, 161, 0, 6212, 0, 15763, 19567
            ),
            init_real_mem(
                89, 5397, 10974, 23091, 260, 6, 11615, 112, 40184, 17656
            ),
            init_real_mem(
                89, 5397, 10974, 23091, 260, 6, 11615, 112, 40184, 17656
            ),
            init_real_mem(
                89, 5397, 10974, 23091, 260, 6, 11615, 112, 40184, 17656
            ),
            init_real_mem(
                89, 5397, 10974, 23091, 261, 2, 11148, 112, 40184, 18244
            ),
            init_real_mem(
                89, 5397, 10974, 23091, 261, 2, 11260, 112, 40184, 19000
            ),
            init_real_mem(
                89, 5397, 10974, 23091, 261, 2, 11372, 112, 40184, 19757
            ),
            init_real_mem(
                89, 5397, 10974, 23091, 261, 2, 11484, 112, 40184, 20513
            ),
            init_real_mem(
                89, 5397, 10974, 23091, 262, 2, 11596, 112, 40184, 21269
            ),
            init_real_mem(
                89, 5397, 10974, 23091, 262, 2, 11708, 112, 40184, 22025
            ),
            init_real_mem(
                89, 5397, 10974, 23091, 262, 2, 11820, 112, 40184, 22782
            ),
            init_real_mem(
                89, 5397, 10974, 23091, 262, 2, 11932, 112, 40184, 23538
            ),
            init_real_mem(
                67, 4048, 8231, 17318, 262, 1, 9577, 112, 30191, 18530
            ),
            init_real_mem(
                67, 4048, 8231, 17318, 262, 2, 9689, 112, 30191, 19333
            ),
            init_real_mem(
                89, 5397, 10974, 23091, 263, 2, 12268, 112, 40184, 25806
            ),
            init_real_mem(
                89, 5397, 10974, 23091, 263, 2, 13139, 112, 40183, 26376
            ),
        ]
        r = test_estim_functions(
            "/mnt/nvme_1_3_4/philippe/mindformers/configs/llama2/hq_ds3_dualpipe_tnd_nd1_fullrec.yaml",
            EVAL,
            real_mem,
            show_res=False,
        )
        res += r

        real_mem = [
            init_real_mem(
                464, 1349, 3628, 9257, 329, 0, 6950, 0, 15520, 32505
            ),
            init_real_mem(
                89, 5397, 10974, 7619, 520, 12, 1191, 224, 24936, 12760
            ),
            init_real_mem(
                89, 5397, 10974, 7619, 520, 12, 1191, 224, 24936, 12760
            ),
            init_real_mem(
                89, 5397, 10974, 7619, 521, 5, 1496, 224, 24936, 13696
            ),
            init_real_mem(
                89, 5397, 10974, 7619, 522, 5, 1608, 224, 24936, 15096
            ),
            init_real_mem(
                89, 5397, 10974, 7619, 522, 5, 1720, 224, 24936, 16497
            ),
            init_real_mem(
                89, 5397, 10974, 7619, 523, 5, 1832, 224, 24936, 17897
            ),
            init_real_mem(
                89, 5397, 10974, 7619, 523, 5, 1944, 224, 24936, 19297
            ),
            init_real_mem(
                89, 5397, 10974, 7619, 524, 5, 2056, 224, 24936, 20698
            ),
            init_real_mem(
                89, 5397, 10974, 7619, 524, 5, 2168, 224, 24936, 22098
            ),
            init_real_mem(
                89, 5397, 10974, 7619, 525, 5, 2280, 224, 24936, 23499
            ),
            init_real_mem(
                89, 5397, 10974, 7619, 525, 5, 2392, 224, 24936, 24899
            ),
            init_real_mem(
                67, 4048, 8231, 5714, 524, 3, 1828, 224, 18748, 19647
            ),
            init_real_mem(
                67, 4048, 8231, 5714, 525, 4, 1940, 224, 18748, 21139
            ),
            init_real_mem(
                89, 5397, 10974, 7619, 527, 5, 2728, 224, 24936, 29101
            ),
            init_real_mem(
                89, 5397, 10974, 7619, 527, 5, 2833, 224, 24935, 30270
            ),
        ]
        r = test_estim_functions(
            "/mnt/nvme_1_3_4/philippe/mindformers/configs/llama2/hq_ds3_dualpipe_tnd_nd2_fullrec.yaml",
            EVAL,
            real_mem,
            show_res=False,
        )
        res += r

        real_mem = [
            init_real_mem(
                464, 1349, 3628, 16372, 624, 0, 3038, 0, 23140, 58298
            ),
            init_real_mem(
                89, 5397, 10974, 8931, 912, 24, 1791, 448, 26765, 21583
            ),
            init_real_mem(
                89, 5397, 10974, 8931, 912, 24, 1791, 448, 26765, 21583
            ),
            init_real_mem(
                89, 5397, 10974, 8931, 912, 11, 1736, 448, 26765, 23383
            ),
            init_real_mem(
                89, 5397, 10974, 8931, 912, 11, 1736, 448, 26765, 26072
            ),
            init_real_mem(
                89, 5397, 10974, 8931, 912, 11, 1736, 448, 26765, 28761
            ),
            init_real_mem(
                89, 5397, 10974, 8931, 912, 11, 1736, 448, 26765, 31450
            ),
            init_real_mem(
                89, 5397, 10974, 8931, 912, 11, 1736, 448, 26765, 34139
            ),
            init_real_mem(
                89, 5397, 10974, 8931, 912, 11, 1736, 448, 26765, 36828
            ),
            init_real_mem(
                89, 5397, 10974, 8931, 912, 11, 1736, 448, 26765, 39516
            ),
            init_real_mem(
                89, 5397, 10974, 8931, 912, 10, 1736, 448, 26765, 42205
            ),
            init_real_mem(
                89, 5397, 10974, 8931, 912, 10, 1736, 448, 26765, 44894
            ),
            init_real_mem(
                67, 4048, 8231, 6698, 912, 6, 1288, 448, 20113, 35348
            ),
            init_real_mem(
                67, 4048, 8231, 6698, 912, 7, 1288, 448, 20113, 38221
            ),
            init_real_mem(
                89, 5397, 10974, 8931, 912, 10, 1736, 448, 26765, 52961
            ),
            init_real_mem(
                89, 5397, 10974, 8931, 912, 10, 1729, 448, 26764, 55419
            ),
        ]
        r = test_estim_functions(
            "/mnt/nvme_1_3_4/philippe/mindformers/configs/llama2/hq_ds3_dualpipe_tnd_nd3_fullrec.yaml",
            EVAL,
            real_mem,
            show_res=False,
        )
        res += r

        real_mem = [
            init_real_mem(
                266, 2698, 5930, 20382, 112, 6, 3934, 0, 31495, 58125
            ),
            init_real_mem(
                89, 5397, 10975, 12486, 912, 475, 3583, 896, 31518, 25800
            ),
            init_real_mem(
                89, 5397, 10975, 12486, 912, 11, 3528, 448, 31518, 30257
            ),
            init_real_mem(
                89, 5397, 10975, 12486, 912, 11, 3528, 448, 31518, 35579
            ),
            init_real_mem(
                89, 5397, 10975, 12486, 912, 11, 3528, 448, 31518, 40901
            ),
            init_real_mem(
                89, 5397, 10975, 12486, 912, 11, 3528, 448, 31518, 46223
            ),
            init_real_mem(
                89, 5397, 10975, 12486, 912, 11, 3528, 448, 31518, 51544
            ),
            init_real_mem(
                78, 4723, 9603, 10925, 912, 9, 3073, 448, 27597, 49960
            ),
        ]
        r = test_estim_functions(
            "/mnt/nvme_1_3_4/philippe/mindformers/configs/llama2/hq_ds3_dualpipe_tnd_nd4_fullrec.yaml",
            EVAL,
            real_mem,
            show_res=False,
        )
        res += r
        print_res(res, "DS3 with DualPipe (512 Die)", plot=PLOT)
        return res

    def test_deepseek_dualpipe_128():
        # 128 Devices
        res = []
        real_mem = [
            init_real_mem(
                277, 3373, 0, 13516, 7600, 2036, 7840, 224, 17446, 34278
            ),
            init_real_mem(
                89, 5397, 0, 9862, 8228, 4884, 2559, 224, 15633, 29089
            ),
        ]
        e = EVAL(
            "/mnt/nvme_1_3_4/philippe/mindformers/configs/llama2/hq_ds3_dualpipe_tnd_128p_nd1_0_0.yaml"
        )
        e.set_passes(swap_os=True)
        r = test_estim_functions(e, None, real_mem, show_res=False)
        res += r
        real_mem = [
            init_real_mem(
                277, 3373, 0, 13516, 4000, 2032, 7191, 0, 17545, 27897
            ),
            init_real_mem(
                89, 5397, 0, 9862, 8228, 4884, 2559, 224, 15633, 29089
            ),
        ]
        e.update_config(
            "/mnt/nvme_1_3_4/philippe/mindformers/configs/llama2/hq_ds3_dualpipe_tnd_128p_nd1_2_0.yaml"
        )
        r = test_estim_functions(e, None, real_mem, show_res=False)
        res += r
        real_mem = [
            init_real_mem(
                277, 3373, 0, 13516, 2800, 2032, 6975, 0, 17545, 25425
            ),
            init_real_mem(
                89, 5397, 0, 9862, 4400, 2973, 2447, 1792, 15730, 22197
            ),
        ]
        e.update_config(
            "/mnt/nvme_1_3_4/philippe/mindformers/configs/llama2/hq_ds3_dualpipe_tnd_128p_nd1_3_3.yaml"
        )
        r = test_estim_functions(e, None, real_mem, show_res=False)
        res += r
        real_mem = [
            init_real_mem(
                277, 3373, 0, 233623, 1600, 2368, 120366, 4032, 237997, 149358
            ),
            init_real_mem(
                89, 5397, 0, 351174, 1600, 3152, 175615, 7169, 357386, 192015
            ),
        ]
        e.update_config(
            "/mnt/nvme_1_3_4/philippe/mindformers/configs/llama2/hq_ds3_dualpipe_tnd_128p_nd1_fullrec_dp64_pp2.yaml"
        )
        r = test_estim_functions(e, None, real_mem, show_res=False)
        res += r
        print_res(res, "DS3 with DualPipe (128 Die)", plot=PLOT)
        return res

    def test_cm2():
        res = []
        test_list = [
            "/mnt/nvme_1_3_4/philippe/cm/mindformers_r150_release/research/yaml/cm_1024p_nd1_dp8_ep8_tp4_pp32_m192.yaml",
            "/mnt/nvme_1_3_4/philippe/cm/mindformers_r150_release/research/yaml/cm_1024p_nd2_dp8_ep8_tp8_pp16_m192.yaml",
            "/mnt/nvme_1_3_4/philippe/cm/mindformers_r150_release/research/yaml/cm_10240p_nd1_dp160_ep8_tp4_pp16_m96.yaml",
            "/mnt/nvme_1_3_4/philippe/cm/mindformers_r150_release/research/yaml/cm_10240p_nd2_dp160_ep8_tp8_pp8_m96.yaml",
            "/mnt/nvme_1_3_4/philippe/cm/mindformers_r150_release/research/yaml/cm_1024p_nd3_ppb_dp32_ep8_tp2_pp16_m48.yaml",
        ]

        real_mems = [
            [
                44058,
                39864,
                38791,
                37638,
                36725,
                35892,
                34812,
                33768,
                32793,
                31672,
                30678,
                29653,
                28853,
                27748,
                26746,
                25642,
                24790,
                23756,
                22783,
                21710,
                20593,
                19635,
                18643,
                17649,
                16738,
                22871,
                21440,
                19877,
                18380,
                16928,
                15392,
                12517,
            ],
            [
                34570,
                31925,
                30807,
                29709,
                28611,
                27533,
                26495,
                25397,
                24299,
                28769,
                27410,
                26087,
                24733,
                23380,
                22042,
                18090,
            ],
            [
                51231,
                46152,
                44217,
                42160,
                40185,
                38199,
                36264,
                34217,
                32223,
                37514,
                35017,
                32582,
                30052,
                27637,
                25128,
                21269,
            ],
            [54272, 51329, 48933, 46531, 44065, 41684, 39262, 34250],
            [
                51675,
                43177,
                41163,
                50983,
                48190,
                45450,
                42818,
                47933,
                47653,
                50293,
                45736,
                41057,
                36596,
                31634,
                27061,
                18896,
            ],
        ]

        def ffn_activ(ccfg, ctx):
            rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
            tok_size = ccfg.s * ccfg.b
            activ_size = (
                ccfg.bytes_compute
                * ccfg.hff
                * (
                    max(ccfg.n_ffMM, ccfg.n_ffBMM)
                    + EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.ffAct)
                    + ccfg.n_ffParamCast
                )
            )
            if ccfg.n_exp > 1:
                return (
                    EvalFFn.shared_exp_activations(ccfg, ctx)
                    + 1.5 * EvalFFn.routed_exp_activations(ccfg, ctx)
                    + EvalFFn.ffn_router_and_concat_activations(ccfg, ctx)
                )
            return activ_size * tok_size

        e = None
        for real, test in zip(real_mems, test_list):
            e = EvaluatorV2(test)
            e.set_ffn_eval_fun(activ=ffn_activ)
            r = test_estim_functions(e, None, real, show_res=False)
            res += r
        print_res(res, "CM 1024+10240 Die", plot=PLOT)
        return res

    def test_xy():
        from memory_estimation.hooks.texthawk_hooks import XY

        res = []
        real_mem = [33120, 20380]
        e = EVAL(
            "test_cases/xy/xy_test1.json",
            hook_cls=XY(),
        )
        r = test_estim_functions(e, None, real_mem, show_res=False)
        res += r
        real_mem = [28768, 20380]
        e.update_config("test_cases/xy/xy_test1_1d.json")
        r = test_estim_functions(e, None, real_mem, show_res=False)
        res += r
        real_mem = [22624, 20380]
        e.update_config("test_cases/xy/xy_test1_2d.json")
        r = test_estim_functions(e, None, real_mem, show_res=False)
        res += r
        real_mem = [16851, 18119]
        e.update_config("test_cases/xy/xy_test1_fullrec.json")
        r = test_estim_functions(e, None, real_mem, show_res=False)
        res += r
        real_mem = [24672, 13218]
        e.update_config("test_cases/xy/xy_test1_16exp.json")
        r = test_estim_functions(e, None, real_mem, show_res=False)
        res += r
        # 128p
        # real_mem = [
        #     1129532,
        #     345765,
        #     376244,
        #     317488,
        #     259928,
        #     201710,
        #     143855,
        #     69636,
        # ]
        # e.update_config(
        #     "/mnt/nvme_1_3_4/philippe/new_XY/MindSpeed-MM-XY/examples/mm_model/texthawk_ds/8p_dryrun/model_norec.json"
        # )
        # r = test_estim_functions(e, None, real_mem, show_res=False)
        # res += r
        # 512p
        real_mem = [
            35044,
            37437,
            46265,
            45554,
            45641,
            44146,
            43539,
            42849,
            41973,
            41347,
            40723,
            40153,
            39158,
            38354,
            37762,
            38720,
        ]
        e.update_config("test_cases/xy/hq.json")
        r = test_estim_functions(e, None, real_mem, show_res=False)
        res += r
        real_mem = [
            42750,
            44608,
            45947,
            46273,
            55785,
            51802,
            51152,
            50368,
            50394,
            48739,
            50705,
            47414,
            47503,
            46263,
            45420,
            46575,
        ]
        e.update_config("test_cases/xy/vpp2.json")
        r = test_estim_functions(e, None, real_mem, show_res=False)
        res += r
        real_mem = [
            46488,
            51073,
            42963,
            51323,
            50908,
            49850,
            49117,
            48891,
            47646,
            47271,
            46339,
            45439,
            44704,
            43969,
            43256,
            44100,
        ]
        e.update_config("test_cases/xy/vpp2_18_ViT.json")
        r = test_estim_functions(e, None, real_mem, show_res=False)
        res += r
        real_mem = [
            46415,
            45995,
            49577,
            51095,
            43735,
            49187,
            48791,
            48587,
            47862,
            47277,
            46429,
            45638,
            45024,
            44436,
            43585,
            44709,
        ]
        e.update_config("test_cases/xy/vpp2_ppb_strat1.json")
        r = test_estim_functions(e, None, real_mem, show_res=False)
        res += r
        print_res(res, "XiaoYi", plot=PLOT)
        return res


def extract_mem_from_rank_logs(dir_path):
    from collections import deque
    import re

    num_lines = 1000  # Last lines
    res = []
    for r, _, l in os.walk(dir_path):
        for f in l:
            p = os.path.join(r, f)
            if p.endswith(".log") and "msrun_log" in p:
                with open(p, "r") as f:
                    lines = list(deque(f, maxlen=num_lines))
                    lines = list(
                        set(
                            filter(lambda x: "actual peak" in x.lower(), lines)
                        )
                    )
                    if not lines:
                        print(
                            f"real peak not found in last {num_lines} of file {p}"
                        )
                        continue
                    res += [
                        (
                            int(re.search(r"(\d+)", lines[0]).group(1)),
                            int(re.search(r"rank_(\d+)", p).group(1)),
                        )
                    ]
    res = [val for val, _ in sorted(res, key=lambda x: x[1])]
    return res


"""Call tests"""
if __name__ == "__main__":
    import inspect
    import argparse

    tests = []  # Test Queue
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", nargs="?")
    parser.add_argument(
        "--comp", help="Show memory insights", action="store_true"
    )
    parser.add_argument(
        "--trace", help="Show evaluation function trace", action="store_true"
    )
    parser.add_argument(
        "--plot", help="Plot memory Prediction vs Real", action="store_true"
    )
    parser.add_argument(
        "--range",
        help="Show only lowest+highest accuracies",
        action="store_true",
    )
    parser.add_argument(
        "--uncertainty", help="Set uncertainty fit threshold value"
    )
    parser.add_argument("--ppb", help="Show PPB output", action="store_true")
    parser.add_argument("--real")
    parser.add_argument("--stage")

    parser.add_argument("--all", action="store_true")
    parser.add_argument("--llama3", action="store_true")
    parser.add_argument("--llama2", action="store_true")
    parser.add_argument("--ds", action="store_true")
    parser.add_argument("--mixtral", action="store_true")
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--cm", action="store_true")
    parser.add_argument("--cm2", action="store_true")
    parser.add_argument("--dual512", action="store_true")
    parser.add_argument("--dual128", action="store_true")
    parser.add_argument("--ds_gmm", action="store_true")
    parser.add_argument("--xy", action="store_true")
    args = parser.parse_args()

    if args.comp:
        VERBOSE_COMP = 1
    if args.trace:
        LAYER_LOG = 1
    if args.plot:
        PLOT = 1
    if args.range:
        RANGE_ONLY = 1
    if args.ppb:
        PPB = 1
    if args.uncertainty:
        UNCERTAINTY = int(args.uncertainty)
    if args.stage:
        SPEC_STAGE_ID = int(args.stage)

    if args.file_path:
        LAYER_LOG = 1
        real_mem = None
        if args.real:  # Either a log path or a list of numeric values
            if os.path.exists(args.real):
                real_mem = extract_mem_from_rank_logs(args.real)
            else:
                real_mem = [
                    int(r.replace("[", "").replace("]", ""))
                    for r in args.real.split(",")
                ]
        test_estim_functions(args.file_path, EVAL, real_mem)
    else:
        if args.all:
            tests = [
                t
                for _, t in inspect.getmembers(
                    Tests, predicate=inspect.isfunction
                )
            ]
        else:
            if args.llama3:
                tests += [Tests.test_llama3]
            if args.llama2:
                tests += [Tests.test_llama2]
            if args.ds:
                tests += [Tests.test_deepseek]
            if args.mixtral:
                tests += [Tests.test_mixtral]
            if args.qwen:
                tests += [Tests.test_qwen]
            if args.cm:
                tests += [Tests.test_cm]
            if args.cm2:
                tests += [Tests.test_cm2]
            if args.dual512:
                tests += [Tests.test_deepseek_dualpipe_512]
            if args.dual128:
                tests += [Tests.test_deepseek_dualpipe_128]
            if args.ds_gmm:
                tests += [Tests.test_deepseek_gmm]
            if args.xy:
                tests += [Tests.test_xy]
            if not tests:
                tests = [
                    t
                    for _, t in inspect.getmembers(
                        Tests, predicate=inspect.isfunction
                    )
                ]

        test_res = []
        for t in tests:
            res = t()
            if res:
                test_res += res
        if test_res:
            preds = [y[3][0] for y in test_res if y[3][0] >= 0]
            reals = [y[3][1] for y in test_res if y[3][1] >= 0]
            logger.info(f"Global MAPE (%) = {mape(preds,reals)}")
            logger.info(f"Global R² = {r2(preds,reals)}")
