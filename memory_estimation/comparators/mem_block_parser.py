import sys
import pprint
import csv
from collections import defaultdict
import re

csv.field_size_limit(500 * 1024 * 1024)


class MemBlockParser:
    """Parser for memory_block.csv"""

    def __init__(self, filename):
        self.filename = filename
        self.stat_lay, self.dyn_lay = {}, {}

    @staticmethod
    def mb(x):
        """bytes to megabytes"""
        return x // 1024 // 1024

    def get_peak(self, memory_changes, need_cumlate=True):
        """search for peak time and memory"""
        if not memory_changes:
            return
        sorted_times = sorted(memory_changes.keys())
        cumulative_memory = 0
        times = []
        memory_usage = []

        for time in sorted_times:
            if need_cumlate:
                cumulative_memory += memory_changes[time]
                times.append(time)
                memory_usage.append(cumulative_memory)
            else:
                times.append(time)
                memory_usage.append(memory_changes[time])

        max_index = memory_usage.index(max(memory_usage))
        peak_time = times[max_index]
        peak_memory_usage = memory_usage[max_index]
        return peak_time, peak_memory_usage

    def track(self):
        """track down peak memory tensors from peak time"""
        objects = []
        real_memory_changes = defaultdict(int)
        actual_memory = defaultdict(int)
        with open(self.filename, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            # real memory
            for row in reader:
                objects.append(row)
                start = int(row["start_time_stamp"])
                end = int(row["end_time_stamp"])
                size = int(row["size"])
                if "actual_peak_memory" in row.keys():
                    actual_peak = int(row["actual_peak_memory"])
                    actual_memory[start] = actual_peak
                if start == -1:
                    continue
                real_memory_changes[start] += size
                real_memory_changes[end] -= size

        real_peak_time, peak_mem = self.get_peak(real_memory_changes)
        _, peak_mem_frag = self.get_peak(actual_memory, False)

        return self.categorize_peak_tensors(
            objects, real_peak_time, peak_mem, peak_mem_frag
        )

    def categorize_peak_tensors(
        self, objects, real_peak_time, peak_mem, peak_mem_frag
    ):
        """categorization"""
        dyn = {}
        stat = {}
        last_visited_node = "others"
        for idx, row in enumerate(objects):
            start = int(row["start_time_stamp"])
            end = int(row["end_time_stamp"])
            if start == -1:
                continue

            lay_id = "others"
            low = row["node_name"].lower()
            lay = re.search(r"(\d+)-.*layer", low)
            if lay:
                lay_id = lay.groups()[0]
            elif "word_embedding" in low:
                lay_id = "emb"
            elif "final_layernorm" in low:
                lay_id = "out"
            if "mtp" in low:
                lay_id = "mtp_" + lay_id
            if "gradients" in low:
                lay_id = "G_" + lay_id
            if last_visited_node != lay_id and "others" not in lay_id:
                last_visited_node = lay_id
            if "other" in lay_id and last_visited_node != "others":
                lay_id = last_visited_node
            if lay_id not in self.dyn_lay:
                self.dyn_lay[lay_id] = {
                    "_activ": 0,
                    "_activ_others": 0,
                    "_activ_attn": 0,
                    "_activ_ffn": 0,
                    "_activ_norm": 0,
                    "ag": 0,
                    "a2a": 0,
                    "ar": 0,
                }
            if start <= real_peak_time <= end:
                if row["type"] != "Kernel":
                    self.categorize_static(stat, row)
                else:
                    self.categorize_dynamic(dyn, row, lay_id)
        return stat, dyn, peak_mem, peak_mem_frag

    def categorize_static(self, stat, row):
        """categorization"""
        low = row["node_name"].lower()
        if row["node_name"] not in stat:
            stat[row["node_name"]] = 0
        stat[row["node_name"]] += int(row["size"])
        lay_id = "others"
        lay = re.search(r"layers\.(\d+)\.", row["node_name"])
        if lay:
            lay_id = lay.groups()[0]
        if "mtp" in low:
            lay_id = "mtp_" + lay_id
        if lay_id not in self.stat_lay:
            self.stat_lay[lay_id] = {"p": 0, "os": 0, "grad": 0}
        if "accu_grad" in row["node_name"]:
            self.stat_lay[lay_id]["grad"] += MemBlockParser.mb(
                int(row["size"])
            )
        elif any(os in row["node_name"] for os in ["adam_", "muon_"]):
            self.stat_lay[lay_id]["os"] += MemBlockParser.mb(
                int(row["size"])
            )
        else:
            self.stat_lay[lay_id]["p"] += MemBlockParser.mb(
                int(row["size"])
            )

    def categorize_dynamic(self, dyn, row, lay_id):
        """categorization"""
        low = row["node_name"].lower()
        if row["node_name"] not in dyn:
            dyn[row["node_name"]] = 0
        dyn[row["node_name"]] += int(row["size"])
        if any(s in low for s in ["alltoall", "all2all"]):
            self.dyn_lay[lay_id]["a2a"] += MemBlockParser.mb(
                int(row["size"])
            )
        elif "allreduce" in low:
            self.dyn_lay[lay_id]["ar"] += MemBlockParser.mb(
                int(row["size"])
            )
        elif "allgather" in low:
            self.dyn_lay[lay_id]["ag"] += MemBlockParser.mb(
                int(row["size"])
            )
        else:
            if "norm" in low:
                self.dyn_lay[lay_id][
                    "_activ_norm"
                ] += MemBlockParser.mb(int(row["size"]))
            elif "attention" in low:
                self.dyn_lay[lay_id][
                    "_activ_attn"
                ] += MemBlockParser.mb(int(row["size"]))
            elif any(
                s in low
                for s in ["feedforward", "mlp", "moe", "router"]
            ):
                self.dyn_lay[lay_id][
                    "_activ_ffn"
                ] += MemBlockParser.mb(int(row["size"]))
            else:
                self.dyn_lay[lay_id][
                    "_activ_others"
                ] += MemBlockParser.mb(int(row["size"]))
            # if lay_id == "others":
            #     print(low , MemBlockParser.mb(int(row["size"])))
            self.dyn_lay[lay_id]["_activ"] += MemBlockParser.mb(
                int(row["size"])
            )

    def summarize(self, stat_mem, dyn_mem, peak_mem, peak_mem_frag):
        """logs"""
        stat = sum(stat_mem.values())
        dyn = sum(dyn_mem.values())
        # pprint.pprint(self.stat_lay, width=200)
        cleaned_dyn_lay = {}
        comm, activ = 0, 0
        for k, v in self.dyn_lay.items():
            cleaned_v = {name: mb for name, mb in v.items() if mb > 0}
            if cleaned_v:
                activ += sum(
                    mb for name, mb in cleaned_v.items() if name == "_activ"
                )
                comm += sum(
                    mb
                    for name, mb in cleaned_v.items()
                    if name in ["ag", "a2a"]
                )
                cleaned_dyn_lay[k] = cleaned_v
                # cleaned_dyn_lay[k].update(self.stat_lay[k])
        for k, v in self.stat_lay.items():
            if k in cleaned_dyn_lay:
                cleaned_dyn_lay[k].update(v)
            else:
                cleaned_dyn_lay[k] = v
        pprint.pprint(cleaned_dyn_lay, width=300)
        print(f"Profiled peak memory: {MemBlockParser.mb(peak_mem)}")
        print(
            f"Profiled peak memory with fragments: "
            f"{MemBlockParser.mb(peak_mem_frag)}"
        )
        print(f"Profiled static memory: {(MemBlockParser.mb(stat))}")
        print(f"Profiled dynamic memory: {(MemBlockParser.mb(dyn))}")
        print("- Profiled total activ:", activ)
        print("- Profiled total comm:", comm)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tracker.py memory_block.csv")
        sys.exit(1)

    file_name = sys.argv[1]
    mbp = MemBlockParser(file_name)
    stat, dyn, peak_mem, peak_mem_frag = mbp.track()
    mbp.summarize(stat, dyn, peak_mem, peak_mem_frag)
