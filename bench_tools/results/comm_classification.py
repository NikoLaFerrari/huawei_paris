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

"""classification of primitives using trace files and IR Graphs"""

import bench_tools.ir.graph as G
from bench_tools.results.bench_result import BenchResult
from bench_tools.results.post_processor import PostProcessor
from bench_tools.utils.ir_utils import (
    get_scope_op_map,
    get_communication_domains,
    get_parallel_dimensions,
)
from bench_tools import ms_trace


@PostProcessor.register
class CommunicationClassifier(PostProcessor):
    """ classify comm primitives using IR Graphs & trace files """
    name = "comm_classification"
    default_config = {}

    def pp_classification(self, op: G.Operator) -> str:
        """
        classfiy PP comm and CP ring-attention 
        """
        if op.type in ["Send", "Receive"]:
            if (
                op.has_prim_attrs()
                and "RING_ATTENTION_INDEX" in op.prim_attrs()
            ) or (
                op.has_cnode_attrs()
                and "RING_ATTENTION_INDEX" in op.cnode_attrs()
            ):
                return ""  # CP ring attention
            return "PP"
        return ""

    def ep_classification(self, op: G.Operator) -> str:
        """
        classify EP comm
        """
        if op.type in ["AllToAll", "AlltoAllV", "AlltoAllVC"]:
            return "EP"
        # In dropless MoE, EP is manual redistribution, find definition of the python instance
        if op.source_file is not None and op.source_file.endswith(
            "mindformers/modules/transformer/moev3.py"
        ):
            return "EP"
        return ""

    def get_depend_source(self, var: G.Variable) -> G.Variable:
        """
        follow Depend chains backward to 
        return first non-Depend source variable
        """
        if not var.is_op():
            return var

        op = var.op()
        if op.type == "Depend":
            return self.get_depend_source(op.inputs[0])
        return var

    def get_depend_users(self, var: G.Variable) -> list[G.Variable]:
        """
        follow Depend chains forward to 
        return all non-Depend lead user variables
        """
        if not var.is_op():
            return [var]

        op = var.op()
        if op.type == "Depend":
            res = []
            for child in op.outputs:
                res += self.get_depend_users(child)
            return res
        return [var]

    def dp_op_classification_by_weight(self, op: G.Operator) -> str:
        """
        classify DP vs OP comm using heurtisitics
        by detecting optimizer update comm patterns
        """
        # op -> (Depend * N) -> Mul(optimizer) -> Depend(%para)
        def is_for_weight_update(op: G.Operator):
            if len(op.outputs) == 1:
                optimizer_vars = self.get_depend_users(op.outputs[0])
                if len(optimizer_vars) == 1 and optimizer_vars[0].is_op():
                    optimizer_op = optimizer_vars[0].op()
                    if optimizer_op.outputs[0].is_op():
                        depend_op = optimizer_op.outputs[0].op()
                        if (
                            depend_op.type == "Depend"
                            and depend_op.inputs[0].is_param()
                        ):
                            return True
            return False

        # %para -> (Depend * N) -> AG
        if op.type == "AllGather":
            source_var = self.get_depend_source(op.inputs[0])
            if source_var.is_param() or (
                isinstance(source_var.op().inputs[0], G.Variable)
                and source_var.op().inputs[0].is_param()
            ):
                return "OP"

        elif op.type == "ReduceScatter":
            if not op.outputs[0].is_op():
                return ""
            if op.outputs[0].op().type == "AllReduce":
                # Partial OP
                # RS -> AR -> (Depend * N) -> Mul(optimizer) -> Depend(%para)
                last_comm = op.outputs[0].op()
            else:
                # RS -> (Depend * N) -> Mul(optimizer) -> Depend(%para)
                last_comm = op
            if is_for_weight_update(last_comm):
                return "OP"

        # AR -> (Depend * N) -> Mul(optimizer) -> Depend(%para)
        elif op.type == "AllReduce":
            if not op.outputs[0].is_op():
                return ""
            if is_for_weight_update(op):
                return "DP"

        return ""

    def dp_op_classification_by_instance_name(self, op: G.Operator) -> str:
        """
        classify DP vs OP
        by using known MindSpore markers
        """
        if op.has_instance_name():
            if "parallel_optimizer_allgather" in op.instance_name():
                return "OP"
            if "grad_mirror_MirrorMicroStepOperator" == op.instance_name():
                return "DP"
        return ""

    def global_norm_classification(self, op: G.Operator) -> str:
        """
        identify global-norm related collectives using markers
        """
        if op.has_instance_name() and op.instance_name() in [
            "PARALLEL_GLOBALNORM_IN_STAGES",
            "PARALLEL_GLOBALNORM_BETWEEN_STAGES",
        ]:
            return "GlobalNorm"
        return ""

    def dataset_classification(self, op: G.Operator) -> str:
        """
        identify Broadcast/Dataset related collectives
        """
        if op.type == "Broadcast":
            return "Dataset"
        if op.type == "AllGather" and op.inputs[0].is_op():
            previous_op = op.inputs[0].op()
            # Broadcast -> Depend -> AG
            if (
                previous_op.type == "Depend"
                and previous_op.inputs[1].is_op()
                and previous_op.inputs[1].op().type == "Broadcast"
            ):
                return "Dataset"
            # Broadcast -> TupleGetItem -> AG
            if (
                previous_op.type == "TupleGetItem"
                and previous_op.inputs[0].is_op()
                and previous_op.inputs[0].op().type == "Broadcast"
            ):
                return "Dataset"
        return ""

    def single_path_to_realdiv(self, op: G.Operator) -> bool:
        """
        return if following the unique output chain 
        eventually reaches a RealDiv op
        """
        if "RealDiv" in op.type:
            return True
        if len(op.outputs) > 1:
            return False
        if not op.outputs[0].is_op():
            return False
        return self.single_path_to_realdiv(op.outputs[0].op())

    def load_balance_classification(self, op: G.Operator) -> str:
        """
        classify LoadBalance related collectives
        """
        if op.type in ["AllGather", "AllReduce"]:
            if self.single_path_to_realdiv(op):
                return "LoadBalance"
        return ""

    def mp_sp_classification(self, op: G.Operator, comm_domains) -> str:
        """
        classify MP vs SP collectives when comm domain is TP only
        """
        if len(comm_domains) == 1 and comm_domains[0] == "TP":
            if op.type in ["AllGather", "ReduceScatter"]:
                return "SP"
            if op.type == "AllReduce":
                return "MP"
        return ""

    def cp_classification(self, op: G.Operator, comm_domains) -> str:
        """
        classify CP primitives and ring-attention Send/Receive markers
        """
        if op.type in ["Send", "Receive"]:
            if (
                op.has_prim_attrs()
                and "RING_ATTENTION_INDEX" in op.prim_attrs()
            ) or (
                op.has_cnode_attrs()
                and "RING_ATTENTION_INDEX" in op.cnode_attrs()
            ):
                return "RingAttention"
        if len(comm_domains) == 1 and comm_domains[0] == "CP":
            return "CP"
        return ""

    def mp_classification_by_group(self, op: G.Operator):
        """
        classfiy MP collectives using primitive group names
        excluding optimizer allgather
        """
        if (
            op.has_instance_name()
            and "parallel_optimizer_allgather" in op.instance_name()
        ):
            return ""
        if (
            op.has_prim_attrs()
            and "group" in op.prim_attrs()
            and "tp-" in op.prim_attrs()["group"]
        ):
            return "MP"
        return ""

    def communication_classification(
        self, op: G.Operator, parallel_dimensions
    ) -> str:
        """
        compute final comm label
        combining multiple heuristics
        returning a fallback class
        """
        # Here we concat classification string so that we know if something is classified multiple times
        classification = ""
        classification += self.pp_classification(op)
        classification += self.mp_classification_by_group(op)
        if classification != "":
            return classification

        classification += self.ep_classification(op)
        classification += self.dp_op_classification_by_instance_name(
            op
        )  # choose one method for dp
        # classification += dp_op_classification_by_weight(op) # choose one method for dp
        classification += self.global_norm_classification(op)
        classification += self.load_balance_classification(op)

        if (
            classification == ""
        ):  # Dataset can have false positives (e.g. AG weight)
            classification += self.dataset_classification(op)
        if (
            classification == ""
        ):  # MP/SP classification can have false positives (e.g. AG tok embedding weight)
            comm_domains = get_communication_domains(op, parallel_dimensions)
            classification += self.mp_sp_classification(op, comm_domains)
            classification += self.cp_classification(op, comm_domains)

        if classification == "":
            classification = "OtherComm"

        return classification

    def mesh_communication_classification(
        self, _, parallel_dimensions
    ):
        """
        placeholder for mesh-based classifucation
        """
        tp = parallel_dimensions["TP"]
        cp = parallel_dimensions["CP"]
        dp = parallel_dimensions["DP"]
        pp = parallel_dimensions["TP"]

        num_devices = tp * cp * dp * pp
        _ = num_devices

    def execute(self, bench_result: BenchResult) -> None:
        """
        populate bench_results.metrics['comm_classfication'] 
        by classifying trace comm events.
        """
        classification = {}
        scope_op_map = get_scope_op_map(bench_result.trace_code_graph)
        parallel_dimensions = get_parallel_dimensions(bench_result.yaml_config)
        communication_pid = ms_trace.find_communication_pid(
            bench_result.process_info
        )
        communication_events = ms_trace.find_communication_events(
            bench_result.process_info, communication_pid
        )
        for event in communication_events:
            if "args" not in event or "mindspore_op" not in event["args"]:
                print(f"mindspore_op not found for {event['name']}")
                continue
            scope = event["args"]["mindspore_op"].removeprefix(
                "Kernel::KernelLaunch::"
            )
            op = scope_op_map[scope]
            classification[scope] = self.communication_classification(
                op, parallel_dimensions
            )
        bench_result.metrics["comm_classification"] = classification
