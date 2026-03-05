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
""" extract per-bucket primtives from trace files & IR Graphs """

import re
import yaml
import json
from bench_tools import prof, ms_trace
from bench_tools.utils.ir_utils import (
    get_largest_graph_from_graph_dir,
    get_scope_op_map,
    get_parallel_dimensions,
)
from bench_tools.results.comm_classification import CommunicationClassifier


class Extractor:
    """ Extractor class """
    def __init__(self, trace_paths, graph_dirs, config_paths):
        self.trace_paths = (
            [trace_paths] if isinstance(trace_paths, str) else trace_paths
        )
        self.graph_dirs = (
            [graph_dirs] if isinstance(graph_dirs, str) else graph_dirs
        )
        self.config_paths = (
            [config_paths] if isinstance(config_paths, str) else config_paths
        )

        self.lane_map = {
            "DP": "DP_COMM",
            "OtherComm": "DP_COMM",
            "OP": "DP_COMM",
            "Optimizer_Internal": "DP_COMM",
            "GlobalNorm": "DP_COMM",
            "MP": "MP_COMM",
            "LoadBalance": "MP_COMM",
            "EP": "EP_COMM",
            "PP": "PP_COMM",
            "Dataset": "BUBBLE",
        }
        self.cd = {}

    def clean(self, name):
        """
        cleans extracted name from trace file.
        """
        if not name:
            return name
        return re.sub(r"-op\d+$", "", name)

    def get_classification(self, process_info, graph_scope_map, parallel_dims):
        """
        takes info of a primitive as input.
        calls bench_tools.comm_classification.CommunicationCalssifier().
        returns the classification bucket the primitive belongs to.
        """
        cd = {
            "COMPUTE": 0,
            "UNKNOWN_COMM": 0,
            "DP_COMM": 0,
            "MP_COMM": 0,
            "PP_COMM": 0,
            "EP_COMM": 0,
            "BUBBLE": 0,
        }
        comm_classifier = CommunicationClassifier()
        comm_pid = ms_trace.find_communication_pid(process_info)
        comm_events = ms_trace.find_communication_events(
                process_info, comm_pid
        )
        for event in comm_events:
            args = event.get("args")
            dur = event.get("dur")
            raw_scope = args.get("mindspore_op").removeprefix(
                    "Kernel::KernelLaunch::"
            )
            op = graph_scope_map.get(raw_scope)
            if not op:
                op = graph_scope_map.get(raw_scope)

            classi = (
                comm_classifier.communication_classification(op, parallel_dims)
                if op
                else "unknown"
            )
            lane = self.lane_map.get(classi)
            if not lane:
                lane = "UNKNOWN_COMM" if classi == "unknown" else "BUBBLE"

            cd[lane] += dur

        comp_pid = ms_trace.find_compute_pid(process_info)
        tid = ms_trace.find_kernels_tid(process_info, comp_pid)
        compute_events = prof.get_thread_events(process_info, comp_pid, tid)
        for event in compute_events:
            dur = event.get("dur")
            name = event.get("name", "").lower()
            if name.lower() in ["event_wait", "event_record"]:
                lane = "BUBBLE"
                cd[lane] += dur
                continue
            lane = "COMPUTE"
            cd[lane] += dur

        total = sum(cd.values())
        pctg = {k: (v / total) * 100 for k, v in cd.items()}
        print(f"[extractor] classification:\n{json.dumps(pctg, indent=2)}")
        self.cd = cd
        return self.cd, pctg

    def run_extractor(self):
        """
        Main Function.
        1. uses bench_tools functions to:
            - parse trace file & ir graphs
            - classify communication & compute events
        2. returns extracted info as dict {bucket: {primitives: {info}}}
        """
        all_classifications = []
        all_samples = []
        comm_classifier = CommunicationClassifier()

        for i in range(len(self.trace_paths)):
            print(f"[regression] Extracting from trace {i + 1}...")
            process_info = prof.parse_process_info(self.trace_paths[i])
            print(f"[regression] Extracting from graph {i + 1}...")
            print(f"[extractor] graph_dir: {self.graph_dirs[i]}")
            graph_obj = get_largest_graph_from_graph_dir(
                self.graph_dirs[i], 0, r"trace_code_graph"
            )
            graph_scope_map = get_scope_op_map(graph_obj)

            with open(self.config_paths[i], "r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
            parallel_dims = get_parallel_dimensions(cfg)

            structured_data = {}

            comm_pid = ms_trace.find_communication_pid(process_info)
            comm_events = ms_trace.find_communication_events(
                    process_info, comm_pid
            )

            for event in comm_events:
                args = event.get("args", {})
                dur = event.get("dur", 0)
                raw_scope = args.get("mindspore_op", "").removeprefix(
                    "Kernel::KernelLaunch::"
                )
                op = graph_scope_map.get(raw_scope)
                if not op:
                    op = graph_scope_map.get(self.clean(raw_scope))

                ir_class = (
                    comm_classifier.communication_classification(
                        op, parallel_dims
                    )
                    if op
                    else "unknown"
                )
                lane = self.lane_map.get(ir_class)
                if not lane:
                    if ir_class == "unknown":
                        lane = "UNKNOWN_COMM"
                    else:
                        lane = "BUBBLE"

                primitive = (
                    self.clean(raw_scope).split("/")[-1]
                    if raw_scope
                    else "UnknownOp"
                )
                bucket_key = f"{lane}::{primitive}"

                size = 0
                if op and op.output_tensors:
                    size = op.output_tensors[0].get_size()

                if not size or size <= 1:
                    count = args.get("count", 0)
                    dtype_size = (
                            2
                            if "16" in str(args.get("data_type", ""))
                            else 4
                    )
                    size = count * dtype_size

                structured_data.setdefault(bucket_key, []).append(
                        {"x": size, "y": dur}
                )

            compute_pid = ms_trace.find_compute_pid(process_info)
            tid = ms_trace.find_kernels_tid(process_info, compute_pid)
            compute_events = prof.get_thread_events(
                    process_info,
                    compute_pid,
                    tid
            )

            for event in compute_events:
                dur = event.get("dur", 0)
                if dur < 0.5:
                    continue  # Sieve out driver noise

                name = event.get("name", "").lower()

                if name in ["event_wait", "event_record"]:
                    structured_data.setdefault("BUBBLE::Wait", []).append(
                        {"x": 1, "y": dur}
                    )
                    continue

                raw_scope = (
                    event["args"]
                    .get("mindspore_op", "")
                    .removeprefix("Kernel::KernelLaunch::")
                )
                if not raw_scope:
                    continue

                op = graph_scope_map.get(raw_scope)
                if not op:
                    op = graph_scope_map.get(self.clean(raw_scope))

                primitive = self.clean(raw_scope).split("/")[-1]
                bucket_key = f"COMPUTE::{primitive}"

                size = 1
                if op and op.output_tensors and len(op.output_tensors) > 0:
                    raw_size = op.output_tensors[0].get_size()
                    size = raw_size if (raw_size and raw_size > 0) else 1

                structured_data.setdefault(bucket_key, []).append(
                        {"x": size, "y": dur}
                )

            all_classifications.append(
                self.get_classification(
                    process_info,
                    graph_scope_map,
                    parallel_dims
                )
            )

            all_samples.append(
                    {"data": structured_data,
                     "dims": parallel_dims}
            )

        return all_samples, all_classifications
