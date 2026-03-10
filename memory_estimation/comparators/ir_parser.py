import re
import pprint
import math
import ast
import argparse


class IRParser:
    def __init__(self, ir_path):
        self.ir_path = ir_path

    @staticmethod
    def byte_for_dtype(x):
        """floating point"""
        if "16" in x:
            return 2
        if "32" in x:
            return 4
        if "64" in x:
            return 8
        if "8" in x:
            return 1
        return 0

    def parse_tracker_graph(self):
        """tracker graph ir"""
        nodes = {}
        visiting_node, visiting_node_idx, path = None, -1, None
        children = []
        idx = 0
        with open(self.ir_path, "r") as f:
            l = f.readline()
            while l:
                q_node = re.search(r"\(([^\)]+)\)\s*=\s*(.*),\s*task", l)
                q_shapes = re.findall(r"(\w+:\[[^\]]*\])(?=.*<-)", l)
                if q_node:
                    groups = q_node.groups()
                    visiting_node = [n.strip() for n in groups[0].split(",")]
                    path = groups[1].split("(")[0]
                    visiting_node_idx = idx
                    children = re.findall(r"(%\d+)", groups[1])
                elif q_shapes:
                    dtypes = [o.split(":")[0] for o in q_shapes]
                    shapes = [
                        tuple(ast.literal_eval(o.split(":")[-1]))
                        for o in q_shapes
                    ]
                    if visiting_node and idx == visiting_node_idx + 1:
                        for i, v in enumerate(visiting_node):
                            if v not in nodes:
                                nodes[v] = {}
                                nodes[v]["path"] = path
                                nodes[v]["shape"] = shapes[i]
                                nodes[v]["dtype"] = dtypes[i]
                                nodes[v]["children"] = children
                    visiting_node, visiting_node_idx, path = None, -1, None
                l = f.readline()
                idx += 1
        return {}, nodes

    def parse_validate(self):
        """valide ir parser"""
        stat, dyn = {}, {}
        visiting_shapes, visiting_shape_idx = None, -1
        idx = 0
        with open(self.ir_path, "r") as f:
            l = f.readline()
            while l:
                q_node = re.search(
                    r"Fullname\s*with\s*scope:\s*\(([^)]*)\)", l
                )
                q_shapes = re.search(r":\s*\(<.*>\)\s*->\s*\(<(.*)>\)", l)
                q_param = re.match(
                    r"^%para\d+_(.*?):\s*<Ref\[Tensor\[(.*)\]\],\s*(.*), ref",
                    l,
                )
                if q_param:
                    group = q_param.groups()
                    if group[0] not in stat:
                        stat[group[0]] = {}
                        shape = ast.literal_eval(group[2])
                        if isinstance(shape, int):
                            shape = (shape,)
                        if shape == ():
                            shape = (0,)
                        stat[group[0]]["shape"] = shape
                        stat[group[0]]["dtype"] = group[1]
                if q_shapes:
                    visiting_shapes = q_shapes.groups()[0]
                    visiting_shape_idx = idx
                elif q_node:
                    if visiting_shapes and idx - visiting_shape_idx <= 2:
                        dtypes = re.findall(
                            r"Tensor\[(.*?)\](?:\*(\d+))?", visiting_shapes
                        )
                        shapes = re.findall(r"(\(\d+.*?\))", visiting_shapes)
                        if dtypes and shapes:
                            path = q_node.groups()[0]
                            if path not in dyn:
                                dyn[path] = {}
                                dyn[path]["dtype"] = sum(
                                    [
                                        [dt[0]]
                                        * (1 if not dt[1] else int(dt[1]))
                                        for dt in dtypes
                                    ],
                                    [],
                                )
                                dyn[path]["shape"] = [
                                    ast.literal_eval(s) for s in shapes
                                ]
                    visiting_shapes, visiting_shape_idx = None, -1
                l = f.readline()
                idx += 1
        return stat, dyn

    def parse(self):
        if "tracker_graph" in self.ir_path:
            return self.parse_tracker_graph()
        elif "validate" in self.ir_path:
            return self.parse_validate()
        return {}, {}

    # Experimental function
    def generate_dot(self, parsed_nodes):
        import graphviz
        dot = graphviz.Digraph(comment='Dependencies')
        dot.graph_attr['rankdir'] = 'LR'  
        limit=1000
        edges = []
        for k,v in parsed_nodes.items():
            if limit<=0: break
            if "Default/data" in v["path"]: continue
            if "StridedSlice" in v["path"]: continue
            dot.node(k, v["path"])
            for c in v["children"]:
                if "Default/data" in parsed_nodes[c]["path"]: continue
                if "StridedSlice" in parsed_nodes[c]["path"]: continue            
                edges += [(k, c)]
            limit-=1
        for x,y in edges:
            dot.edge(x,y)
        dot_path = self.ir_path.replace("\\","/")
        dot_path = dot_path.split("/")[-1].replace(".ir",".dot")
        print(dot, dot_path)
        dot.render(dot_path)

def main():
    parser = argparse.ArgumentParser(description="IR file parser")
    parser.add_argument(
        "file_path",
        nargs=1,
        help="tracker_graph.ir or validate.ir",
    )
    parser.add_argument(
        "--graph", action="store_true", help="Generate DOT file"
    )
    args = parser.parse_args()
    ir = args.file_path[0]
    if not ir.endswith(".ir"):
        raise argparse.ArgumentTypeError(f"`{ir}` has invalid file type")

    irp = IRParser(ir)
    if "tracker_graph" in ir:
        _, nodes = irp.parse_tracker_graph()
        # pprint.pprint(nodes)
    elif "validate" in ir:
        stat, dyn = irp.parse_validate()
        pprint.pprint(stat, width=200, compact=True)
        pprint.pprint(dyn, width=200, compact=True)
    if args.graph:
        irp.generate_dot(nodes)

if __name__ == "__main__":
    main()
