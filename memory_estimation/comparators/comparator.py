import argparse
import pprint
import sys
import time
import re
import math
from memory_estimation.comparators.mem_block_parser import MemBlockParser
from memory_estimation.comparators.ir_parser import IRParser
from memory_estimation.estimate_v2 import EvaluatorV2

class Comparator:
    def __init__(self, stage_id, paths):
        cfg = next((p for p in paths if p.endswith(".yaml")), None)
        tracker_graph_ir_path = next(
            (p for p in paths if "tracker_graph" in p and p.endswith(".ir")),
            None,
        )
        validate_ir_path = next(
            (p for p in paths if "validate" in p and p.endswith(".ir")), None
        )
        mem_block_path = next((p for p in paths if p.endswith(".csv")), None)
        if not cfg or not mem_block_path:
            print(f"Missing files cfg:{cfg}, mem_block {mem_block_path}")
            exit(0)
        self.e = EvaluatorV2(cfg, log_level=0)
        # self.e.trace_formula("layer_activations")
        self.e.print_ccfg()
        self.irp, self.irp2 = None, None
        if validate_ir_path:
            self.irp = IRParser(validate_ir_path)
        if tracker_graph_ir_path:
            self.irp2 = IRParser(tracker_graph_ir_path)
        self.mbp = MemBlockParser(mem_block_path)
        self.align = 512
        self.stage = stage_id

    def green(self, x):
        return f"\033[92m{x}\033[00m"

    def red(self, x):
        return f"\033[91m{x}\033[00m"

    def yellow(self, x):
        return f"\033[93m{x}\033[00m"

    def bold(self, x):
        return f"\033[1m{x}\033[00m"

    def shorten(self, path):
        split = path.split("/")
        lay = next(
            (s for s in split if re.search(r"\d+-TransformerLayer", s)), None
        )
        if len(split) > 2:
            return "/".join(
                [(split[0] if not lay else lay), "...", split[-2], split[-1]]
            )
        return path

    def retrieve_ir_nodes_from_path(self, nodes, path):
        res = []
        for name, node in nodes.items():
            if name == path or ("path" in node and node["path"] == path):
                res += [node]

        # Unique
        return [dict(s) for s in set(frozenset(d.items()) for d in res)]
        # uniq_res = []
        # for r in res:
        #     if not any(x.values()==r.values() for x in uniq_res):
        #         uniq_res += [r]
        # return uniq_res

    def compare_mem(self, mem, total):
        if abs(total - mem) == 0:
            # print(f"\t{self.bold(mem)} ({MemBlockParser.mb(mem)} MB) = {self.green(total)}")
            return 0
        elif abs(total - mem) == self.align:
            # print(f"\t{self.bold(mem)} ({MemBlockParser.mb(mem)} MB) = " + self.green(f"{total} + {self.align} (align)"))
            return self.align
        return -1

    def compare(self):
        print(f"\n{'='*10} Tracking peak... {'='*10}")
        start = time.time()
        stat_mem, dyn_mem, peak, peak_frag = self.mbp.track()
        end = time.time()
        stat_ir, dyn_ir, stat_ir2, dyn_ir2 = [None] * 4
        print(f"Elapsed time: {end - start} seconds")
        if self.irp:
            print(f"\n{'='*10} Parsing IR (1/2)... {'='*10}")
            start = time.time()
            stat_ir, dyn_ir = self.irp.parse()
            end = time.time()
            print(f"Elapsed time: {end - start} seconds")
            print(f"\n{'='*10} Comparing static... {'='*10}")
            self.compare_mem_trace(stat_mem, stat_ir)
            # self.compare_mem_trace(dyn_mem, dyn_ir)
        if self.irp2:
            print(f"\n{'='*10} Parsing IR (2/2)... {'='*10}")
            start = time.time()
            stat_ir2, dyn_ir2 = self.irp2.parse()
            end = time.time()
            print(f"Elapsed time: {end - start} seconds")
            print(f"\n{'='*10} Comparing dynamic... {'='*10}")
            self.compare_mem_trace(dyn_mem, dyn_ir2)
        if not self.irp and not self.irp2:
            for path, mem in stat_mem.items():
                mb = MemBlockParser.mb(mem)
                if mb > 0:
                    print(
                        f"{self.bold(MemBlockParser.mb(mem))}  \t\t{self.shorten(path)}"
                    )
            for path, mem in dyn_mem.items():
                mb = MemBlockParser.mb(mem)
                if mb > 0:
                    print(
                        f"{self.bold(MemBlockParser.mb(mem))} \t\t{self.shorten(path)}"
                    )
        print()
        self.compare_mem_prediction(
            stat_mem, dyn_mem, peak, peak_frag
        )  # to do: for all stages

    def compare_mem_prediction(self, stat_mem, dyn_mem, peak, peak_frag):
        insights = self.e.estimate_peak_insight()
        print(f"\n{'='*10} Predicted {'='*10}")
        if self.stage >= 0:
            # self.e.estimate_peak(spec_stage_id=self.stage, verbose=True)
            node_logs = insights[self.stage]
            pprint.pprint(node_logs["Node Log"], width=300)
            comm = sum(mb for name, mb in node_logs.items() if "Comm" in name)
            activ = node_logs["Dynamic"] - comm
            print(
                f"Predicted peak memory: {node_logs['Static']+node_logs['Dynamic']}"
            )
            print("Predicted static memory:", node_logs['Static'])
            print("Predicted dynamic memory:", node_logs['Dynamic'])
            print("- Predicted total activ:", activ)
            print("- Predicted total comm:", comm)
            # pprint.pprint(node_logs)
        print(f"\n{'='*10} Profiled {'='*10}")
        self.mbp.summarize(stat_mem, dyn_mem, peak, peak_frag)

    def compare_mem_trace(self, mem_insight, ir_source):
        ignored, ignored_mem = 0, 0
        for path, mem in mem_insight.items():
            if mem <= self.align:
                # print(f'{self.shorten(path)} IGNORED, {self.bold(mem)} ({MemBlockParser.mb(mem)} MB) <= {self.align} (align)')
                ignored += 1
                ignored_mem += mem
                continue
            if ignored > 0:
                print(
                    self.yellow(
                        f"IGNORED {ignored}x mem <= {self.align} (align), total ignored mem: {ignored_mem} ({MemBlockParser.mb(ignored_mem)} MB)"
                    )
                )
                ignored, ignored_mem = 0, 0
            q_nodes = self.retrieve_ir_nodes_from_path(ir_source, path)
            if q_nodes:
                # print(q_nodes)
                # print(self.yellow(f'{self.shorten(path)} found in IR'))
                if self.irp and "validate" in self.irp.ir_path and all(
                    isinstance(q, list) for q in q_nodes[0].values()
                ):
                    dtypes = q_nodes[0]["dtype"]
                    shapes = q_nodes[0]["shape"]
                else:
                    dtypes = [n["dtype"] for n in q_nodes]
                    shapes = [n["shape"] for n in q_nodes]
                byte = [IRParser.byte_for_dtype(d) for d in dtypes]
                # print(f"\tdtypes {dtypes} => {byte}")
                sprods = [
                    s if isinstance(s, int) else math.prod(s) for s in shapes
                ]
                # print(f"\tshapes {shapes} => {prods}")
                prods = [d * s for d, s in zip(byte, sprods)]
                # print(f"\t=> dtypes * shapes {prods}")
                total = sum(prods)
                # print(f"\t=> total {total}\n")
                found = False
                k = self.compare_mem(mem, total)
                count_byte = ",".join(
                    list(
                        set(
                            f"{b}{f'*{byte.count(b)}' if byte.count(b)>1 else ''}"
                            for b in byte
                        )
                    )
                )
                count_shapes = ",".join(
                    list(
                        set(
                            f"{b}{f'*{shapes.count(b)}' if shapes.count(b)>1 else ''}"
                            for b in shapes
                        )
                    )
                )
                count_prods = ",".join(
                    list(
                        set(
                            f"{b}{f'*{prods.count(b)}' if prods.count(b)>1 else ''}"
                            for b in prods
                        )
                    )
                )
                if k < 0:
                    for i, p in enumerate(prods):
                        k = self.compare_mem(mem, p)
                        if k >= 0:
                            found = True
                            # print(f"\t>>> matching with dimension[{i}]")
                            if k == 0:
                                print(
                                    f"{self.shorten(path)}:\n\t{self.bold(mem)} ({MemBlockParser.mb(mem)} MB) = {self.green(p)} (dimension[{i}])"
                                    f"\n\tDT:[{count_byte}]] x S:[{count_shapes}] = [{count_prods}]"
                                )
                            else:
                                print(
                                    f"{self.shorten(path)}:\n\t{self.bold(mem)} ({MemBlockParser.mb(mem)} MB) = {self.green(f'{p} + {k}')} (dimension[{i}]+align)"
                                    f"\n\tDT:[{count_byte}] x S:[{count_shapes}] = [{count_prods}]"
                                )
                            break
                else:
                    found = True
                    # print(f"\t>>> matching with total")
                    if k == 0:
                        print(
                            f"{self.shorten(path)}:\n\t{self.bold(mem)} ({MemBlockParser.mb(mem)} MB) = {self.green(total)} (total)"
                            f"\n\tDT:[{count_byte}] x S:[{count_shapes}] = [{count_prods}]"
                        )
                    else:
                        print(
                            f"{self.shorten(path)}:\n\t{self.bold(mem)} ({MemBlockParser.mb(mem)} MB) = {self.green(f'{total} + {k}')} (total+align)"
                            f"\n\tDT:[{count_byte}] x S:[{count_shapes}] = [{count_prods}]"
                        )
                if not found:
                    # print(f"\tno traces for {self.bold(mem)} ({MemBlockParser.mb(mem)} MB)")
                    print(
                        f"{self.shorten(path)}:\n\t{self.bold(mem)} ({MemBlockParser.mb(mem)} MB) couldn't compute from IR trace"
                        f"\n\tDT:[{count_byte}] x S:[{count_shapes}] = [{count_prods}]"
                    )
            else:
                # print(self.red(f"\t{self.shorten(path)} Not found in IR"))
                # print(f"\tno traces for {self.bold(mem)} ({MemBlockParser.mb(mem)} MB)")
                print(
                    f"{self.shorten(path)}:\n\t{self.bold(mem)} ({MemBlockParser.mb(mem)} MB) {self.red('no trace in IR')}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Command line usage: Estimate peak stage memory"
    )
    parser.add_argument("file_path", nargs="+", help="MF cfg, IRs, Mem CSVs")
    parser.add_argument(
        "--stage", default=-1, type=int, help="Specify pipeline stage ID"
    )
    args = parser.parse_args()

    cmp = Comparator(args.stage, args.file_path)
    cmp.compare()


if __name__ == "__main__":
    main()
