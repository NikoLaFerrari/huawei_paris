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

"""parses IR Graphs into graph-operator-tensor objects """

import re
import ast
from ast import literal_eval as make_tuple

from . import graph as G


def peek_line(file) -> str:
    """ return the next line without advancing the file cursor """
    pos = file.tell()
    line = file.readline()
    file.seek(pos)
    return line


def parse_tensor(tensor_str) -> G.Tensor:
    """ parse a single tensor descriptor string into a Tensor object """
    ref_str = None
    type_delim_idx = tensor_str.find(",")
    type_str = tensor_str[:type_delim_idx]

    if "NoShape" in tensor_str or "null" in tensor_str:
        shape = None
    elif "Tuple" in tensor_str:
        shape = None
    else:
        end_shape_delim = tensor_str.find(")", type_delim_idx)
        shape_str = tensor_str[
            type_delim_idx + 1 : end_shape_delim + 1
        ].strip()

        try:
            shape = ast.literal_eval(shape_str)
        except (ValueError, SyntaxError):
            print(f"WARNING: Could not parse tensor string {shape_str}")
            return None

        if not isinstance(shape, tuple):
            shape = (shape,)

        if type_str.startswith("Ref"):
            ref_str = tensor_str[end_shape_delim + 2 :].strip()
            ref_str = ref_str.split("=")[1].strip()

    return G.Tensor(type=type_str, shape=shape, ref=ref_str)


def parse_tensor_list(tensor_list_str) -> list[G.Tensor]:
    """ parse a paranthesized list of tensor descriptors into Tensor objects """
    if tensor_list_str[0] != "(" or tensor_list_str[-1] != ")":
        raise ValueError(f"Got wrong tensor list format: {tensor_list_str}")

    idx = 0

    str_list = []

    while True:
        beg = tensor_list_str.find("<", idx)
        if beg == -1:
            break

        end = tensor_list_str.find(">", beg)
        tensor_str = tensor_list_str[beg + 1 : end]
        str_list.append(tensor_str)
        idx = end

    tensor_list = []
    for tensor_str in str_list:
        tensor = parse_tensor(tensor_str)
        tensor_list.append(tensor)

    return tensor_list


def find_scope_close(txt, start_idx, end_idx, open_symb, close_symb):
    """ 
    find the matching closing symbol index for a scoped region 
    nesting supported
    """
    nesting = 0
    for i in range(start_idx, end_idx):
        if txt[i] == open_symb:
            nesting += 1
        elif txt[i] == close_symb:
            nesting -= 1

        if nesting == 0:
            return i

    return -1


def find_input_separators(line, sep, start_idx, end_idx):
    """ 
    return indices of separators at top-level scope 
    ignores nested (), [], {}
    """
    sep_idxs = []
    scope_stack = []

    scope_tokens = {
        "{": "}",
        "(": ")",
        "[": "]",
    }

    for i in range(start_idx, end_idx):
        c = line[i]
        if len(scope_stack) == 0 and c == sep:
            sep_idxs.append(i)
        elif c in scope_tokens.keys():
            scope_stack.append(c)
        elif len(scope_stack) > 0 and c == scope_tokens[scope_stack[-1]]:
            del scope_stack[-1]

    return sep_idxs


def remove_type_info(input_str):
    """ strip IR scalar type wrappers from a string """
    type_list = ["I64", "Bool", "F32", "F16", "U64"]

    for type_name in type_list:
        idx = 0

        while True:
            idx = input_str.find(type_name, idx)

            if idx == -1:
                break

            # remove type name and oppening parenthesis
            input_str = input_str[:idx] + input_str[idx + len(type_name) + 1 :]

            # remove corresponding closing parenthesis
            idx = input_str.find(")", idx)
            input_str = input_str[:idx] + input_str[idx + 1 :]

    return input_str


def parse_op_input(line, start_idx, end_idx, subgraph):
    """ parse a single operator inpyt """
    if line.startswith("%", start_idx, end_idx):
        return G.Variable(line[start_idx:end_idx], subgraph)

    if line.startswith("@", start_idx, end_idx):
        return G.GraphReference(line[start_idx:end_idx])

    if "Tensor" in line[start_idx:end_idx]:
        return G.RawValue("Tensor")

    input_str = line[start_idx:end_idx]
    input_str = remove_type_info(input_str)

    return G.RawValue(input_str)


def split_list_scopes(line, sep, start_idx, end_idx):
    """ split a scoped list """
    input_list = []
    sep_idxs = find_input_separators(line, sep, start_idx, end_idx)
    sep_i = 0
    prev_sep_idx = start_idx
    for sep_i in range(0, len(sep_idxs) + 1):
        end_sep_idx = sep_idxs[sep_i] if sep_i < len(sep_idxs) else end_idx

        input_list.append(line[prev_sep_idx:end_sep_idx])

        prev_sep_idx = end_sep_idx
        if prev_sep_idx == end_idx:
            break
        prev_sep_idx += 2

    return input_list


def extract_op_input_list(line, start_idx, end_idx, subgraph) -> list:
    """ extract and parse operator inputs """
    input_list = []
    sep_idxs = find_input_separators(line, ",", start_idx, end_idx)
    sep_i = 0
    prev_sep_idx = start_idx
    for sep_i in range(0, len(sep_idxs) + 1):
        end_sep_idx = sep_idxs[sep_i] if sep_i < len(sep_idxs) else end_idx

        if line[prev_sep_idx:end_sep_idx]:
            input_list.append(
                parse_op_input(line, prev_sep_idx, end_sep_idx, subgraph)
            )

        prev_sep_idx = end_sep_idx
        if prev_sep_idx == end_idx:
            break
        prev_sep_idx += 2

    return input_list


def parse_tuple_with_key(line, key):
    """ extract and eval a tuple value following a dict-style key in a string """
    start_idx = line.find(key)
    if start_idx == -1:
        return None

    end_idx = find_scope_close(line, start_idx + len(key), len(line), "(", ")")
    ret_str = line[start_idx + len(key) : end_idx + 1]
    return make_tuple(ret_str)


def parse_device_matrix(line):
    """ parse the device_matrix tuple """ 
    return parse_tuple_with_key(line, "'device_matrix': ")


def parse_tensor_map(line):
    """ parse the tensor_map tuple """
    return parse_tuple_with_key(line, "'tensor_map': ")


def parse_tensor_layout(layouts_str) -> list[G.TensorLayout]:
    """ parse tensor layout into TensorLayout objects"""
    end_idx = find_scope_close(layouts_str, 0, len(layouts_str), "(", ")")

    tensor_layouts = []

    layout_strs = split_list_scopes(layouts_str, ",", 1, end_idx)

    for layout_str in layout_strs:
        device_matrix = parse_device_matrix(layout_str)
        tensor_map = parse_tensor_map(layout_str)

        layout = G.TensorLayout(device_matrix, tensor_map)
        tensor_layouts.append(layout)

    return tensor_layouts


def parse_tensor_line(file):
    """ parse the `: (<...> -> (<...>)` tensor line """
    # Parse input tensor list
    line = file.readline()

    line = line[7:].strip()
    in_out_sep = "->"
    sep_idx = line.find(in_out_sep)

    inputs_str = line[:sep_idx].strip() if sep_idx != -1 else line
    input_tensors = parse_tensor_list(inputs_str)

    if sep_idx != -1:
        outputs_str = line[sep_idx + len(in_out_sep) :].strip()
        output_tensors = parse_tensor_list(outputs_str)
    else:
        outputs_str = ""
        output_tensors = None

    # skip extra tensor lines
    next_line = peek_line(file)
    while next_line is not None and next_line.strip().startswith(": "):
        file.readline()
        next_line = peek_line(file)

    return input_tensors, output_tensors


def parse_special_attr_vals(attrs) -> dict:
    """ convert certain attribute strings into Python objects """
    if "in_strategy" in attrs.keys():
        attrs["in_strategy"] = ast.literal_eval(attrs["in_strategy"])

    if "out_strategy" in attrs.keys():
        attrs["out_strategy"] = ast.literal_eval(attrs["out_strategy"])

    if "primitive_attrs" in attrs.keys():
        if "in_strategy" in attrs["primitive_attrs"].keys():
            attrs["primitive_attrs"]["in_strategy"] = ast.literal_eval(
                attrs["primitive_attrs"]["in_strategy"]
            )
        if "out_strategy" in attrs["primitive_attrs"].keys():
            attrs["primitive_attrs"]["out_strategy"] = ast.literal_eval(
                attrs["primitive_attrs"]["out_strategy"]
            )
        if "group_rank_ids" in attrs["primitive_attrs"].keys():
            attrs["primitive_attrs"]["group_rank_ids"] = ast.literal_eval(
                attrs["primitive_attrs"]["group_rank_ids"]
            )

        if "in_layout" in attrs["primitive_attrs"]:
            attrs["primitive_attrs"]["in_layout"] = parse_tensor_layout(
                attrs["primitive_attrs"]["in_layout"]
            )

    return attrs


def parse_attr_dict_str(attrs_str) -> dict:
    """ parse attribute dict into a Python dict """
    attr_dict = {}

    attrs_list = split_list_scopes(attrs_str, ",", 0, len(attrs_str))

    for attr in attrs_list:
        key_sep_idx = attr.index(":")
        attr_key = attr[:key_sep_idx].strip()
        attr_val = remove_type_info(attr[key_sep_idx + 1 :]).strip()
        attr_val = attr_val.replace('"', "")

        attr_dict[attr_key] = attr_val

    return attr_dict


def parse_op_attributes(line, start_pos, end_pos) -> dict:
    """ parse attribute dicts in op line into a merged dict """
    attrs = {}

    idx = start_pos
    while True:
        dict_start_pos = line.find("{", idx, end_pos)
        if dict_start_pos == -1:
            break

        dict_key = line[idx : dict_start_pos - 2].strip()

        dict_end_pos = find_scope_close(
            line, dict_start_pos, end_pos, "{", "}"
        )

        dict_str = line[dict_start_pos + 1 : dict_end_pos]

        attr_val = parse_attr_dict_str(dict_str)

        if dict_key == "":
            for key, val in attr_val.items():
                attrs[key] = val
        else:
            attrs[dict_key] = attr_val

        idx = dict_end_pos + 1

    attrs = parse_special_attr_vals(attrs)

    return attrs


def parse_scope(file) -> str:
    """ parse the `# Fullname with scope: (...)` line if present """
    line = file.readline()
    if not line:
        return None
    line = line.strip()

    start_pattern = "# Fullname with scope: ("

    beg = line.find(start_pattern) + len(start_pattern)

    if beg == -1:
        return None

    end = line.find(")", beg)
    if end == -1:
        return None

    return line[beg:end]


def parse_source_file(file) -> str:
    """ 
    parse the most recent `# In file ...` and 
    return the source file path 
    """
    start_pattern = "# In file "
    last_file_line = None

    while True:
        next_line = peek_line(file)
        if not next_line:
            return None

        next_line = next_line.strip()

        if not next_line.startswith("# "):
            if last_file_line is None:
                return None
            last_file_line = last_file_line.removeprefix(start_pattern)
            last_file_line = last_file_line[: last_file_line.find(" ")]
            return last_file_line[: last_file_line.rfind(":")]

        if next_line.startswith(start_pattern):
            last_file_line = next_line

        file.readline()


def parse_op(file, subgraph: G.SubGraph) -> G.Operator:
    """ parse one operator """
    line = None

    while True:
        line = file.readline()
        if not line:
            return None

        line = line.strip()
        if line == "":
            return None

        if line.startswith("%") or line.startswith("Return"):
            break
    tensor_pattern = re.compile(r": \(.*\)")

    while True:
        # to be compatible with line breaking of the first line
        next_line = peek_line(file)
        if tensor_pattern.match(next_line.strip()):
            break
        line += file.readline().strip()

    idx = 0

    op_idx = ""
    #var_name = None

    if line.startswith("%"):
        end_op_idx = line.index("(")
        op_idx = line[:end_op_idx].strip()
        idx = end_op_idx
        var_name_end = find_scope_close(line, idx, len(line), "(", ")")
        #var_name = line[idx + 1 : var_name_end]
        idx = var_name_end + 1
        idx += 3  # Skip ' = '

    if line.startswith("%", idx):
        op_type_end = line.find("]", idx)
        if op_type_end == -1:
            op_type_end = line.find(")", idx)

        op_type_end += 1
    else:
        op_type_end = line.find("(", idx)

    op_type = line[idx:op_type_end].strip()

    idx = op_type_end
    input_list_end = find_scope_close(line, idx, len(line), "(", ")")

    inputs = extract_op_input_list(line, idx + 1, input_list_end, subgraph)

    idx = input_list_end + 2

    attrs = parse_op_attributes(line, idx, len(line))

    input_tensors, output_tensors = parse_tensor_line(file)
    scope = parse_scope(file)
    source_file = parse_source_file(file)

    op = G.Operator(
        idx=op_idx,
        type=op_type,
        inputs=inputs,
        attrs=attrs,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        scope=scope,
        source_file=source_file,
        subgraph=subgraph,
    )

    return op


def parse_ops(file, subgraph: G.SubGraph) -> None:
    """ parse all operators for a subgraph until EOF """
    subgraph.ops = {}

    while True:
        op = parse_op(file, subgraph)

        if op is None:
            break

        op.subgraph = subgraph
        subgraph.ops[op.idx] = op


def parse_params(file, _: G.Graph) -> None:
    """ skip parsing params section """
    # Skip params as it's not needed for now

    while True:
        line = file.readline()
        if not line or line.strip() == "":
            break


def add_op_users(subgraph: G.SubGraph) -> None:
    """ 
    for each op, populate `outputs` list 
    by linking Varible inputs into producing ops
    """
    for op in subgraph.ops.values():
        for op_input in op.inputs:
            if (
                isinstance(op_input, G.Variable)
                and op_input.key in subgraph.ops
            ):
                subgraph.ops[op_input.key].outputs.append(
                    G.Variable(op.idx, subgraph)
                )


def build_id_links(subgraph: G.SubGraph) -> None:
    """ build fw/bw unique-id links between ops in a subgraph """
    for op in subgraph.ops.values():
        unique_id = op.unique_id()
        if unique_id is None:
            continue

        subgraph.unique_id_to_idx[unique_id] = op.idx

    for op in subgraph.ops.values():
        if op.has_forward_unique_id():
            if op.forward_unique_id() not in subgraph.unique_id_to_idx:
                continue

            op.f_op_idx = subgraph.unique_id_to_idx[op.forward_unique_id()]
            f_op = subgraph.ops[op.f_op_idx]
            if f_op.b_ops_idx is None:
                f_op.b_ops_idx = []
            f_op.b_ops_idx.append(op.idx)

        if op.has_unique_id() and op.b_ops_idx is None:
            op.b_ops_idx = []


def skip_empty_lines(file) -> None:
    """ advance file cursor past any empty lines """
    while True:
        next_line = peek_line(file)
        if not next_line:
            break
        if next_line.strip() != "":
            break
        file.readline()


def parse_subgraphs(file, graph: G.Graph) -> None:
    """ parse each subgraph instance and append it to the graph """
    while True:
        skip_empty_lines(file)

        line = file.readline()
        if not line:
            break

        line = line.strip()

        if "subgraph instance: " in line:
            name = line.split(":")[1].strip()

            subgraph = G.SubGraph()
            subgraph.name = name

            skip_empty_lines(file)
            parse_ops(file, subgraph)

            add_op_users(subgraph)
            build_id_links(subgraph)

            graph.subgraphs.append(subgraph)


def parse_header(file, graph: G.Graph) -> None:
    """ parse graph header """
    line = file.readline()
    graph.entry = line.split(":")[1].strip()

    line = file.readline()  # Total subgraphs

    while True:  # Attrs
        line = file.readline()
        if not line or line.strip() == "":
            break


def parse_graph(path: str) -> G.Graph:
    """ parse graph from path to Graph object """
    graph = G.Graph()
    with open(path, "r", encoding='utf-8') as file:
        parse_header(file, graph)
        parse_params(file, graph)
        parse_subgraphs(file, graph)
    check_input_shapes(graph)
    check_op_refs(graph)

    return graph


def check_op_refs(graph: G.Graph) -> None:
    """ placeholder for validating op references """
    for subgraph in graph.subgraphs:
        for op in subgraph.ops.values():
            for _  in op.inputs:
                pass


def check_input_shapes(graph: G.Graph) -> None:
    """ warn if the number of parsed op inputs dont match tensor signature """
    for subgraph in graph.subgraphs:
        for op in subgraph.ops.values():
            inputs = len(op.inputs)
            input_tensors = len(op.input_tensors)

            if inputs != input_tensors:
                print(
                    f"WARNING: input and input tensors are not "
                    f"the same for op {op.idx} ({op.type}) in "
                    f"subgraph {subgraph.name}"
                )
                print(
                    f"Got inputs {op.inputs} ({inputs}) and "
                    f"input_tensors {op.input_tensors} ({input_tensors})"
                )

                for inp in op.inputs:
                    print("input", inp)
