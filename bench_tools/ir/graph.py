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

"""identifies events/primitives found in the trace in the IR Graphs"""


class Variable:
    """ reference a value in a subgraph """
    def __init__(self, key, subgraph):
        self.key: str = key
        self.subgraph: SubGraph = subgraph

    def __str__(self):
        return self.key

    def __repr__(self):
        return str(self)

    def __eq__(self, value):
        return (
            isinstance(value, Variable)
            and self.key == value.key
            and self.subgraph.name == value.subgraph.name
        )

    def is_param(self) -> bool:
        """ ireturn if this variable is a graph parameter """
        return self.key.startswith("%para")

    def is_op(self) -> bool:
        """ 
        returns if this variable key corresponds 
        to an operator id in the subgraph 
        """
        return self.key in self.subgraph.ops

    def op(self) -> "Operator":
        """ 
        return the Operator produced by this variable if it is an op output,
        else None 
        """
        # return the corresponding operator of the Varaible
        return self.subgraph.ops[self.key] if self.is_op() else None


class GraphReference:
    """ reference to another graph """
    def __init__(self, key):
        self.key = key

    def __str__(self):
        return "@" + self.key

    def __repr__(self):
        return str(self)


class RawValue:
    """ unparsed value appearing in IR inputs/attrs """
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return f"{self.text}"

    def __repr__(self):
        return str(self)


class Tensor:
    """ Tensor type metadata """
    def __init__(self, type, shape, ref):
        self.shape: tuple[int] = shape
        self.type: str = type
        self.ref: str = ref

    def get_size(self) -> int:
        """ 
        return tensor size in bytes 
        None if shape unkonwn
        -1 if any dimension is dynamic
        """
        if self.shape is None:
            return None

        unit = get_unit_of_datatype(self.type)
        size = 1
        for i in self.shape:
            size *= i
            if i == -1:  # Dynamic shape
                return -1
        return size * unit


def get_unit_of_datatype(datatype: str) -> int:
    """ 
    returns the number of bytes per element for 
    a given IR datatype string 
    """
    if datatype.startswith("Ref[") and datatype.endswith("]"):
        datatype = datatype[4:-1]
    if datatype.startswith("Tensor[") and datatype.endswith("]"):
        datatype = datatype[7:-1]
    if datatype in ["Bool", "Int8", "UInt8"]:
        return 1
    if datatype in ["Float16", "BFloat16", "Int16", "UInt16"]:
        return 2
    if datatype in ["Float32", "Int32", "UInt32"]:
        return 4
    if datatype in ["Float64", "Int64", "UInt64"]:
        return 8
    print(f"Warning: Datatype {datatype} not found")
    return 1


def shapes_to_str(tensors):
    """ 
    convert a list of Tensor objects into a compact string of their shapes
    """
    if tensors is None:
        return "None"

    shapes_str = "("
    first = True
    for _, tensor in enumerate(tensors):
        if tensor is None or not tensors.shape:
            continue

        if not first:
            shapes_str += ", "

        shapes_str += str(tensor.shape)
        first = False
    shapes_str += ")"

    return shapes_str


def layout_to_stra(op: "Operator"):
    """ 
    convert operator's input tensor layouts into a strategy matrix form
    """
    stra = []

    for layout in op.in_layout():
        tensor_stra = []
        for key in layout.tensor_map:
            if key == -1:
                tensor_stra.append(1)
            else:
                tensor_stra.append(layout.device_matrix[-key - 1])

        stra.append(tensor_stra)

    return stra


def stra_to_str(stra) -> str:
    """ 
    convert a strategy into a printable string form
    """
    if stra is None:
        return ""

    stra_str = str(stra)
    stra_str = stra_str.replace("(", "[")
    stra_str = stra_str.replace(")", "]")

    return stra_str


def op_inputs_to_str(op: "Operator") -> str:
    """ pretty-print operator inputs """
    inputs_str = "("
    first = True
    for inp in op.inputs:
        if not first:
            inputs_str += ", "
        first = False
        inputs_str += str(inp)
    inputs_str += ")"

    return inputs_str


class Operator:
    """ Operation node in subgraph """
    def __init__(
        self,
        idx,
        type,
        attrs=None,
        input_tensors=None,
        output_tensors=None,
        inputs=None,
        scope=None,
        source_file=None,
        subgraph=None,
    ):
        self.idx: str = idx
        self.type: str = type
        self.attrs: dict = attrs
        self.input_tensors: list[Tensor] = input_tensors
        self.output_tensors: list[Tensor] = output_tensors
        self.inputs: list = inputs
        self.scope: str = scope
        self.source_file: str = source_file
        self.subgraph: SubGraph = subgraph
        self.outputs: list[Variable] = []
        self.f_op_idx: str = None
        self.b_ops_idx: list[str] = None

    def has_prim_attrs(self):
        """ returns if this operator has 'primitive_attrs' dict """
        return (
            self.attrs is not None and "primitive_attrs" in self.attrs.keys()
        )

    def prim_attrs(self):
        """ returns primtive_attrs dict, else None """
        return self.attrs["primitive_attrs"] if self.has_prim_attrs() else None

    def has_cnode_attrs(self):
        """ returns if operator has 'cnode_attrs' dict """
        return self.attrs is not None and "cnode_attrs" in self.attrs.keys()

    def cnode_attrs(self):
        """ returns cnode_attrs, else None """
        return self.attrs["cnode_attrs"] if self.has_cnode_attrs() else None

    def has_cnode_prim_attrs(self):
        """ returns if operator has cnode_prim_attrs dict """
        return (
            self.attrs is not None
            and "cnode_primal_attrs" in self.attrs.keys()
        )

    def cnode_prim_attrs(self):
        """ returns cnode_prim_attrs, else None """
        return (
            self.attrs["cnode_primal_attrs"]
            if self.has_cnode_prim_attrs()
            else None
        )

    def has_prim_stra(self):
        """ returns if operator has prim_stra dict  """
        return (
            self.has_prim_attrs() and "in_strategy" in self.prim_attrs().keys()
        )

    def has_auto_stra(self):
        """ returns if operator has auto_stra """
        return self.attrs is not None and "in_strategy" in self.attrs

    def has_stra(self):
        """ returns if operator has stra """
        return self.has_auto_stra() or self.has_prim_stra()

    def prim_stra(self):
        """ return prim_stra, else None """
        return (
            self.prim_attrs()["in_strategy"] if self.has_prim_stra() else None
        )

    def auto_stra(self):
        """ returns auto_stra, else None """
        return self.attrs["in_strategy"] if self.has_auto_stra() else None

    def has_in_layout(self):
        """ returns if operator has in_layout dict """
        return (
            self.has_prim_attrs() and "in_layout" in self.prim_attrs().keys()
        )

    def in_layout(self):
        """ return in_layout dict, else None """
        return self.prim_attrs()["in_layout"] if self.has_in_layout() else None

    def has_instance_name(self):
        """ returns if operator has instance_name """
        return self.attrs is not None and "instance name" in self.attrs.keys()

    def instance_name(self):
        """ returns instance_name """
        return (
            self.attrs["instance name"] if self.has_instance_name() else None
        )

    def is_redistribution(self):
        """ returns if operator is a redistribution op based on name """
        return (
            "redistribution_op" in self.instance_name()
            if self.has_instance_name()
            else False
        )

    def is_communication(self):
        """ 
        returns if this operator type is 
        a known p2p/collective communication op 
        """
        return self.type in [
            "AllGather",
            "AllReduce",
            "ReduceScatter",
            "AlltoAll",
            "AlltoAllV",
            "AlltoAllVC",
            "Broadcast",
            "Send",
            "Receive",
        ]

    def has_unique_id(self):
        """ returns if operator has unique_id """
        return (
            self.has_cnode_prim_attrs()
            and "unique_id" in self.cnode_prim_attrs().keys()
        )

    def unique_id(self):
        """ returns unique_id, else None """
        return (
            self.cnode_prim_attrs()["unique_id"]
            if self.has_unique_id()
            else None
        )

    def has_duplicated(self):
        """ returns if operator has duplicated """
        return (
            self.has_cnode_attrs()
            and "duplicated" in self.cnode_attrs().keys()
        )

    def duplicated(self):
        """ returns duplicated, else None """
        return (
            self.cnode_attrs()["duplicated"] if self.has_duplicated() else None
        )

    def has_related_node_id(self):
        """ returns if operator has related_node_id """
        return (
            self.has_cnode_prim_attrs()
            and "related_node_id" in self.cnode_prim_attrs().keys()
        )

    def related_node_id(self):
        """ returns related_node_id, else None """
        return (
            self.cnode_prim_attrs()["related_node_id"]
            if self.has_related_node_id()
            else None
        )

    def has_forward_unique_id(self):
        """ returns if operator has forward_unique_id """
        return (
            self.has_cnode_prim_attrs()
            and "forward_unique_id" in self.cnode_prim_attrs().keys()
        )

    def forward_unique_id(self):
        """ returns forward_unique_id, else None """
        return (
            self.cnode_prim_attrs()["forward_unique_id"]
            if self.has_forward_unique_id()
            else None
        )

    def has_related_fusion_key(self):
        """ returns if operator has related_fusion_key """
        return (
            self.has_cnode_prim_attrs()
            and "related_fusion_key" in self.cnode_prim_attrs().keys()
        )

    def related_fusion_key(self):
        """ returns related_fusion_key, else None """
        return (
            self.cnode_prim_attrs()["related_fusion_key"]
            if self.has_related_fusion_key()
            else None
        )

    def has_mirror_user_id(self):
        """ returns if operator has mirror_user_id """
        return (
            self.has_cnode_prim_attrs()
            and "mirror_user_id" in self.cnode_prim_attrs().keys()
        )

    def mirror_user_id(self):
        """ returns mirror_user_id, else None """
        return (
            self.cnode_prim_attrs()["mirror_user_id"]
            if self.has_mirror_user_id()
            else None
        )

    def has_related_comm_node_id(self):
        """ returns if operator has related_comm_node_id """
        return (
            self.has_cnode_attrs()
            and "related_comm_node_id" in self.cnode_attrs().keys()
        )

    def related_comm_node_id(self):
        """ returns related_comm_node_id, else None """
        return (
            self.cnode_attrs()["related_comm_node_id"]
            if self.has_related_comm_node_id()
            else None
        )

    def is_forward(self):
        """ returns if this op is a fw op """
        return self.has_unique_id()

    def is_backward(self):
        """ retrns if this op is a bw op """
        return self.has_forward_unique_id()

    def is_recompute(self):
        """
        returns if this op is marked duplicated and 
        its scope contains recompute marker 
        """
        return (
            self.has_duplicated()
            and self.scope is not None
            and isinstance(self.scope, str)
            and self.scope.find("recompute_Default") > -1
        )

    def stra(self):
        """ returns the best available strategy representation """
        if self.has_auto_stra():
            return self.auto_stra()
        if self.has_in_layout():
            return layout_to_stra(self)
        if self.has_prim_stra():
            return self.prim_stra()
        return None

    def __str__(self):
        id_str = (
            f"{self.idx} "
            if self.idx is not None and self.type != "Return"
            else ""
        )

        out_shape_txt = f" -> {shapes_to_str(self.output_tensors)}"

        ret_str = (f"{id_str}{self.type}{op_inputs_to_str(self)} "
                   f"{shapes_to_str(self.input_tensors)}{out_shape_txt} "
                   f"{stra_to_str(self.stra())}")
        return ret_str

    def __repr__(self):
        return str(self)


class TensorLayout:
    """ Tensor sharding layout from device_matrix & tensor_map """
    def __init__(self, device_matrix, tensor_map):
        self.device_matrix = device_matrix
        self.tensor_map = tensor_map

    def __str__(self):
        return f"{{d: {self.device_matrix}, m: {self.tensor_map}}}"

    def __repr__(self):
        return str(self)


class SubGraph:
    """ subgraph class """
    def __init__(self):
        self.name: str = None
        self.ops: dict[str, Operator] = {}
        self.unique_id_to_idx: dict[str, str] = {}


class Graph:
    """ graph class """
    def __init__(self):
        self.params = {}
        self.subgraphs: list[SubGraph] = []
