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

"""per-device parsing and analysis"""

import os
import json
import re
import glob
from functools import cached_property
import yaml
import pandas as pd

from bench_tools.utils.base_utils import logger
from bench_tools.prof import parse_process_info
from bench_tools.ir.parser import parse_graph
import bench_tools.ir.graph as G


class BenchResult:
    """
    - Helper class to retrive results of a single device \\
    - If its a bench_tools runs, use rank=None to automatically \\
    use smallest rank, or specify the rank number \\
    - If the directory structure is different from \\
    bench_tools, please specify the file paths \\
    - DO NOT modified the returned objects of @cached_property !!!
    """

    def __init__(
        self,
        bench_dir: str = None,  # parent dir of .log files
        rank: int = None,
        yaml_config_path: str = None,  # file path
        graph_dir: str = None,  # parent dir of .ir files
        profiler_output_dir: str = None,  # parent dir of trace_view.json
        memory_block_dir: str = None,  # parent dir of memory_block.csv
    ) -> None:
        self.bench_dir = bench_dir
        self.rank = rank
        self.metrics = {}

        self.yaml_config_path = yaml_config_path
        self.graph_dir = graph_dir
        self.profiler_output_dir = profiler_output_dir
        self.memory_block_dir = memory_block_dir

        if self.yaml_config_path is None:
            self.yaml_config_path = self._get_dir(
                os.path.join(self.bench_dir, f"config_{self.hash}.yaml")
            )

        if self.graph_dir is None:
            self._init_graph_path()

        if (
            not self.profiler_output_dir
            or not self.memory_block_dir
            and bench_dir is not None
        ):
            self._init_profile_dir()

    def _init_graph_path(self):
        """
        set self.graph_dir based on YAML config or default graph/ layout
        """
        if "save_graphs_path" in self.yaml_config["context"]:
            save_graphs_path = self.yaml_config["context"]["save_graphs_path"]
            if save_graphs_path.startswith("/"):
                base_graph_dir = save_graphs_path
            else:
                base_graph_dir = os.path.join(self.bench_dir, save_graphs_path)
        else:
            base_graph_dir = os.path.join(self.bench_dir, "graph")
        graph_dir = self._get_rank_dir(base_graph_dir)
        self.graph_dir = self._get_dir(graph_dir)

    def get_all_worker_log_paths(bench_dir):
        """ return sorted worker log filenames found under bench_dir """
        log_path_pattern = os.path.join(bench_dir, "worker_*.log")
        log_paths = glob.glob(log_path_pattern)
        log_paths = [d.split("/")[-1] for d in log_paths]
        log_paths = sorted(
            log_paths,
            key=lambda d: int(d.removeprefix("worker_").removesuffix(".log")),
        )
        return log_paths

    def get_all_ranks(bench_dir):
        """ 
        return sorted integer ranks inferred from worker_*.log under bench_dir
        """
        log_path_pattern = os.path.join(bench_dir, "worker_*.log")
        log_paths = glob.glob(log_path_pattern)
        log_paths = [d.split("/")[-1] for d in log_paths]
        ranks = sorted(
            [
                int(d.removeprefix("worker_").removesuffix(".log"))
                for d in log_paths
            ]
        )
        return ranks

    def _get_rank_dir(self, parent_dir: str) -> str:
        """
        return teh rank subdir path under parent_dir
        """
        if self.rank is not None:
            return os.path.join(parent_dir, f"rank_{self.rank}")
        rank_path_pattern = os.path.join(parent_dir, "rank_*")
        rank_dirs = glob.glob(rank_path_pattern)
        rank_dirs = [d.split("/")[-1] for d in rank_dirs]
        rank_dirs = sorted(rank_dirs, key=lambda d: int(d.split("_")[-1]))
        if len(rank_dirs) == 0:
            logger.warning(
                f"No matching directory found for {rank_path_pattern}"
            )
            return None
        return os.path.join(parent_dir, rank_dirs[0])  # smallest rank

    def _init_profile_dir(self):
        """
        infer and set profiler_output_dir and memory_block_dir 
        from teh profiling layout
        """
        base_profile_dir = os.path.join(self.bench_dir, "output", "profile")
        rank_dir = self._get_rank_dir(base_profile_dir)
        if rank_dir is None:
            return
        profile_dir_pattern = os.path.join(rank_dir, "*_ascend_ms")
        matching_profile_dirs = glob.glob(profile_dir_pattern)
        if len(matching_profile_dirs) == 0:
            return
        if len(matching_profile_dirs) > 1:
            logger.warning(
                f"More than 1 profiling directory detected "
                f"({len(matching_profile_dirs)}). Taking first one"
            )
        if self.profiler_output_dir is None:
            profiler_output_dir = os.path.join(
                matching_profile_dirs[0], "ASCEND_PROFILER_OUTPUT"
            )
            self.profiler_output_dir = self._get_dir(profiler_output_dir)
        if self.memory_block_dir is None:
            framework_dir = os.path.join(matching_profile_dirs[0], "FRAMEWORK")
            memory_block_dir = self._get_rank_dir(framework_dir)
            self.memory_block_dir = self._get_dir(memory_block_dir)

    def _get_dir(self, directory):
        """
        return dir if it exists and is a string, else None
        """
        if not isinstance(directory, str):
            return None
        if os.path.exists(directory):
            return directory
        return None

    # ============ bench_dir ============

    @cached_property
    def run_cfg(self) -> dict:
        """
        load and cache run_cfg.json from bench_dir as dict
        """
        cfg_path = os.path.join(self.bench_dir, "run_cfg.json")
        with open(cfg_path, "r", encoding='utf-8') as fp:
            return json.load(fp)

    @cached_property
    def bench_cfg(self) -> dict:
        """
        load and cache bench_cfg.json from bench_dir as dict
        """
        cfg_path = os.path.join(self.bench_dir, "bench_cfg.json")
        with open(cfg_path, "r", encoding='utf-8') as fp:
            return json.load(fp)

    @cached_property
    def hash(self) -> str:
        """
        return the run hash from run_cfg
        """
        return self.run_cfg.get("hash")

    @cached_property
    def yaml_config(self) -> dict:
        """
        load and cache the YAML config
        """
        with open(self.yaml_config_path, "r", encoding='utf-8') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    @cached_property
    def worker_log_path(self) -> str:
        """
        return the resolved path to the worker log for this rank
        """
        if self.rank is not None:
            return os.path.join(self.bench_dir, f"worker_{self.rank}.log")
        log_paths = BenchResult.get_all_worker_log_paths(self.bench_dir)
        if len(log_paths) == 0:
            print("Warning: No worker log found")
            return None
        return os.path.join(self.bench_dir, log_paths[0])  # smallest rank

    # ============ graph ============

    def get_largest_graph(self, pattern) -> G.Graph:
        """
        find the largest IR graph file in graph_dir matching 
        a base name pattern and parse it
        """
        pattern = pattern + r"_\d+(\.ir)?$"
        matching_graph_list = [
            os.path.join(self.graph_dir, f)
            for f in os.listdir(self.graph_dir)
            if re.search(pattern, f)
        ]
        if len(matching_graph_list) == 0:
            raise FileNotFoundError(f"No ir graph found with pattern {pattern}")
        graph_path = sorted(matching_graph_list, key=os.path.getsize)[-1]
        return parse_graph(graph_path)

    @cached_property
    def trace_code_graph(self) -> G.Graph:
        """
        return the parsed largest trace_code_graph_* IR graph for this run
        """
        return self.get_largest_graph(r"trace_code_graph")

    @cached_property
    def execute_graph(self) -> G.Graph:
        """
        return the parsed longest *_execute_* IR graph for this run
        """
        return self.get_largest_graph(r"\d+_execute")

    # ============ profiler_output ============

    @cached_property
    def process_info(self) -> dict:
        """
        return parsed process info from trace_view.json 
        in the profiler output dir
        """
        if self.profiler_output_dir is None:
            return None
        trace_path = os.path.join(self.profiler_output_dir, "trace_view.json")
        return parse_process_info(trace_path)

    @cached_property
    def operator_memory(self) -> pd.DataFrame:
        """ 
        load operator_memory.csv from profiler_output_dir 
        into a pandas DataFrame
        """
        mem_path = os.path.join(
            self.profiler_output_dir, "operator_memory.csv"
        )
        return pd.read_csv(mem_path)

    # ============ memory_block ============

    @cached_property
    def memory_block(self) -> pd.DataFrame:
        """ 
        load memory_block.csv from memory_block_dir into a pandas DataFrame
        """
        mem_path = os.path.join(self.memory_block_dir, "memory_block.csv")
        return pd.read_csv(mem_path)
