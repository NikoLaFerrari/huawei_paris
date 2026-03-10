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

"""utils methods for results/"""

import os
import glob
import re
import yaml
import pandas as pd
from bench_tools.utils.base_utils import logger, get_profile_path, get_rank_dirs
from bench_tools.prof import parse_process_info


def open_config(bench_id, res_dir="results"):
    """ 
    load the benchmark YAML config for a given run
    """
    with open(
        f'{res_dir}/{bench_id}/config_{bench_id.split("_")[1]}.yaml',
        encoding='utf-8'
    ) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_log_paths(bench_dir):
    """
    return sorted worker log paths under a benchmark directory
    """
    log_path_pattern = os.path.join(bench_dir, "worker_*.log")
    log_paths = glob.glob(log_path_pattern)
    log_paths = sorted(
        log_paths, key=lambda d: int(d.split("_")[-1].split(".")[0])
    )

    return log_paths


def get_bench_graph_from_graph_dir(
    base_graph_dir, graph_pattern, rank: int = 0
):
    """
    return sorted IR Graph paths under an explicit graph dir
    """
    rank_dirs = get_rank_dirs(base_graph_dir)
    if len(rank_dirs) <= rank:
        raise ValueError(
            f"Asked for local rank {rank} while only "
            f"{(rank_dirs)} local ranks are available "
            f"in graph dir {base_graph_dir}"
        )
    rank_dir = rank_dirs[rank]
    graph_dir = os.path.join(base_graph_dir, rank_dir)
    graph_pattern = graph_pattern + r"_\d+(\.ir)?$"
    matching_graph_list = [
        os.path.join(graph_dir, f)
        for f in os.listdir(graph_dir)
        if re.search(graph_pattern, f)
    ]
    matching_graph_list = sorted(
        matching_graph_list, key=lambda f: int(f.split("_")[-1].split(".")[0])
    )

    return matching_graph_list


def get_bench_graph(bench_dir, graph_pattern, rank: int = 0):
    """
    return sorted IR Graph under a benchmark directory
    """
    base_graph_dir = os.path.join(bench_dir, "graph")
    rank_dirs = get_rank_dirs(base_graph_dir)
    if len(rank_dirs) <= rank:
        raise ValueError(
            f"Asked for local rank {rank} "
            f"while only {len(rank_dirs)} local ranks "
            f"are available in graph dir {base_graph_dir}"
        )
    rank_dir = rank_dirs[rank]
    graph_dir = os.path.join(base_graph_dir, rank_dir)
    graph_pattern = graph_pattern + r"_\d+(\.ir)?$"
    matching_graph_list = [
        os.path.join(graph_dir, f)
        for f in os.listdir(graph_dir)
        if re.search(graph_pattern, f)
    ]
    matching_graph_list = sorted(
        matching_graph_list, key=lambda f: int(f.split("_")[-1].split(".")[0])
    )

    return matching_graph_list


def get_memory_block(bench_id, res_dir="results", rank: int = 0):
    """
    load the memory_block.csv DataFrame for a run
    if not present => return None
    """
    bench_dir = os.path.join(res_dir, bench_id)
    profile_path = get_profile_path(bench_dir, rank)
    mem_path_pattern = os.path.join(
        profile_path, "FRAMEWORK/rank_*/memory_block.csv"
    )
    matching_mem_path_list = glob.glob(mem_path_pattern)
    if len(matching_mem_path_list) == 0:
        return None
    if len(matching_mem_path_list) > 1:
        logger.info(
            f"More than 1 memory_block.csv detected "
            f"({len(matching_mem_path_list)}). Taking first one"
        )
    data = pd.read_csv(matching_mem_path_list[0])
    return data


def get_operator_memory(bench_id, res_dir="results", rank: int = 0):
    """
    load operator_memory.csv as pandas.DataFrame
    """
    bench_dir = os.path.join(res_dir, bench_id)
    profile_path = get_profile_path(bench_dir, rank)
    mem_path = os.path.join(
        profile_path, "ASCEND_PROFILER_OUTPUT/operator_memory.csv"
    )
    data = pd.read_csv(mem_path)
    return data


def get_process_info(bench_id, res_dir="results", rank: int = 0):
    """
    parse and return process info from trace_view.json for a run
    """
    bench_dir = os.path.join(res_dir, bench_id)
    profile_path = get_profile_path(bench_dir, rank)
    trace_path = os.path.join(
        profile_path, "ASCEND_PROFILER_OUTPUT/trace_view.json"
    )
    return parse_process_info(trace_path)


def get_layer_key(bench_id, res_dir):
    """
    return the model-specific layer key suffix used for naming decode layers
    """
    layer_key_dict = {
        "DeepseekV3Config": "-DeepSeekV2DecodeLayer",
        "LlamaConfig": "-LLamaDecodeLayer",
    }
    config = open_config(bench_id, res_dir)
    return layer_key_dict[config["model"]["model_config"]["type"]]


def get_layer_name(bench_id, res_dir, layer):
    """
    build the canonical layer name string for a given layer index
    """
    layer_key = get_layer_key(bench_id, res_dir)
    return str(layer) + layer_key
