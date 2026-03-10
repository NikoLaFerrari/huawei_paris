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

"""util methods for IR Graphs"""

import os
import bench_tools.ir.graph as G
from bench_tools.ir.parser import parse_graph
from bench_tools.utils.results_utils import get_bench_graph, get_bench_graph_from_graph_dir


def get_largest_graph_from_graph_dir(graph_dir, rank, pattern) -> G.Graph:
    """
    find the largest IR Graph under graph_dir matching pattern for rank
    and parse it
    """
    graph_paths = get_bench_graph_from_graph_dir(graph_dir, pattern, rank)
    if len(graph_paths) < 1:
        raise FileNotFoundError(
            f"No ir graph found with pattern {pattern} "
            f"for rank {rank} in bench dir {graph_dir}"
        )
    graph_path = sorted(graph_paths, key=os.path.getsize)[-1]
    return parse_graph(graph_path)


def get_largest_graph(bench_id, res_dir, rank, pattern) -> G.Graph:
    """
    find the largest IR Graph file for a given benchmark run and parse it
    """
    bench_dir = os.path.join(res_dir, bench_id)
    graph_paths = get_bench_graph(bench_dir, pattern, rank)
    if len(graph_paths) < 1:
        raise FileNotFoundError(
            f"No ir graph found with pattern {pattern} "
            f"for rank {rank} of {bench_id}"
        )
    graph_path = sorted(graph_paths, key=os.path.getsize)[-1]
    return parse_graph(graph_path)


def get_execute_graph(bench_id, res_dir, rank) -> G.Graph:
    """
    load the largest execute graph for the given benchmark
    """
    return get_largest_graph(bench_id, res_dir, rank, r"\d+_execute")


def get_trace_code_graph(bench_id, res_dir, rank) -> G.Graph:
    """
    load the largest trace-code graph for the given benchmark
    """
    return get_largest_graph(bench_id, res_dir, rank, r"trace_code_graph")


def get_forward_subgraph(execute_graph: G.Graph) -> G.SubGraph:
    """
    return the forward subgraph from an execute graph
    """
    for subgraph in execute_graph.subgraphs:
        if "mindformers_model" in subgraph.name:
            return subgraph
    return None

def get_backward_subgraph(execute_graph: G.Graph) -> G.SubGraph:
    """
    return the backward subgraph from an execute graph
    """
    first = True
    for subgraph in execute_graph.subgraphs:
        if "mindformers_model" in subgraph.name:
            if first:
                first = False
            else:
                return subgraph
    return None


def get_scope_op_map(parsed_graph: G.Graph) -> dict[str, G.Operator]:
    """
    build a mapping from scope string to Operator 
    for all operators in the graph
    """
    ops = {}
    for subgraph in parsed_graph.subgraphs:
        for op in subgraph.ops.values():
            ops[op.scope] = op
    return ops


def get_coords(
    rank, dimensions, parallel_order=["TP", "CP", "DP", "PP"]
) -> dict[str, int]:
    """
    :param rank: Device ID
    :param dimensions: for example {'TP': 2, 'CP': 2, 'DP': 4, 'PP': 2}
    :param parallel_order: the order parallelization of machines, from fine to corse grain
    :return: coords on all dimensions
    """
    coords = {}

    for name in parallel_order:
        coords[name] = rank % dimensions[name]
        rank //= dimensions[name]
    return coords


def get_communication_domains(
    op: G.Operator, dimensions, parallel_order=["TP", "CP", "DP", "PP"]
) -> list[str]:
    """
    :param op: Operator
    :param dimensions: for example {'TP': 2, 'CP': 2, 'DP': 4, 'PP': 2}
    :param parallel_order: the order parallelization of machines, from fine to corse grain
    :return: list of communication domain
    """
    if "group_rank_ids" in op.prim_attrs():
        group_rank_ids = op.prim_attrs()["group_rank_ids"]
        if isinstance(group_rank_ids, tuple):
            coords = [
                get_coords(rank, dimensions, parallel_order)
                for rank in group_rank_ids
            ]
            varying_dims = []

            for dim in parallel_order:
                if not all(
                    c[dim] == coords[0][dim] for c in coords
                ):  # if more than one rank in this dimension
                    varying_dims.append(dim)

            return varying_dims

    return []


def get_parallel_dimensions(
    config,
) -> dict[str, int]:  # for coords and communication domains
    """
    extract TP-CP-DP-PP sizes from a benchmark YAML config dict
    """
    return {
        "TP": config["parallel_config"].get("model_parallel", 1),
        "CP": config["parallel_config"].get("context_parallel", 1),
        "DP": config["parallel_config"].get("data_parallel", 1),
        "PP": config["parallel_config"].get("pipeline_stage", 1),
    }
