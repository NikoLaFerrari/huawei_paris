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

"""parses through trace files and idenitfies primitives"""

import json


def create_trace_pid(process_info):
    """ return teh next available pid for process_info """
    return len(process_info.keys())


def create_trace_tid(process_info, pid):
    """ return the next available tid for a given pid """
    return len(process_info[pid]["threads"].keys())


def add_process_sort_index(process_info, pid, sort_index):
    """ set the process-level sort_index metadata """ 
    pid_info = get_trace_pid_info(process_info, pid)
    pid_info["sort_index"] = sort_index


def add_thread_sort_index(process_info, pid, tid, sort_index):
    """ set the thread-level sort_index metadata """
    tid_info = get_trace_tid_info(process_info, pid, tid)
    tid_info["sort_index"] = sort_index


def create_trace_process(process_info, name, pid=None):
    """ create a new process entry and return its pid """
    if pid is None:
        pid = create_trace_pid(process_info)
    process_info[pid] = {"name": name, "sort_index": pid, "threads": {}}
    return pid


def create_trace_thread(process_info, name, pid, tid=None):
    """ create a new thread entry under pid and return its tid """
    if tid is None:
        tid = create_trace_tid(process_info, pid)
    process_info[pid]["threads"][tid] = {
        "name": name,
        "sort_index": tid,
        "events": [],
    }
    return tid


def create_event(name, ts, dur, args={}):
    """ create a trace-style complete event dict """ 
    return {"name": name, "ts": ts, "dur": dur, "args": args}


def create_trace_event(process_info, pid, tid, name, ts, dur, args={}):
    """ append a new event to the given process/thread in process_info """
    get_trace_tid_info(process_info, pid, tid)["events"].append(
        create_event(name, ts, dur, args)
    )


def get_trace_pid_info(process_info, pid):
    """ return pid info dict or raise if pid is missing """
    if pid not in process_info.keys():
        raise ValueError(f"PID {pid} does not exist in process info")
    return process_info[pid]


def get_trace_tid_info(process_info, pid, tid):
    """ return tid info dict or raise if tid is missing """
    pid_info = get_trace_pid_info(process_info, pid)
    if tid not in pid_info["threads"].keys():
        raise ValueError(
            f"TID {tid} if PID {pid} does not exist in process info"
        )

    return pid_info["threads"][tid]


def parse_trace_metadata(process_info, data):
    """ populate process/thread metadata from trace 'M' records """
    for d in data:
        if d["ph"] == "M":
            if d["name"] == "process_name":
                create_trace_process(process_info, d["args"]["name"], d["pid"])
            if d["name"] == "process_sort_index":
                process_info[d["pid"]]["sort_index"] = d["args"]["sort_index"]
            if d["name"] == "process_labels":
                process_info[d["pid"]]["labels"] = d["args"]["labels"]
            if d["name"] == "thread_name":
                create_trace_thread(
                    process_info, d["args"]["name"], d["pid"], d["tid"]
                )
            if d["name"] == "thread_sort_index":
                process_info[d["pid"]]["threads"][d["tid"]]["sort_index"] = d[
                    "args"
                ]["sort_index"]


def parse_trace_events(process_info, data):
    """ populate event lists from trace 'X' records """
    for d in data:
        if d["ph"] == "X":
            args = d["args"] if "args" in d.keys() else {}
            create_trace_event(
                process_info,
                d["pid"],
                d["tid"],
                d["name"],
                float(d["ts"]),
                float(d["dur"]),
                args,
            )


def get_process_threads(process_info, pid):
    """ return threads dict for a pid """
    return process_info[pid]["threads"]


def get_thread_events(process_info, pid, tid):
    """ return event list for a given pid/tid """
    return process_info[pid]["threads"][tid]["events"]


def set_thread_events(process_info, pid, tid, events):
    """ replace the event list for a given pid/tid """
    process_info[pid]["threads"][tid]["events"] = events


def parse_process_info(prof_file):
    """ 
    parse a trace file into 
    normalized process/threadsd/event dicts 
    """
    with open(prof_file, encoding='utf-8') as f:
        data = json.load(f)

    process_info = {}
    parse_trace_metadata(process_info, data)
    parse_trace_events(process_info, data)

    return process_info


def find_process_id(process_info, name):
    """ find pid by process name, or return None """
    for pid in process_info:
        if process_info[pid]["name"] == name:
            return pid
    return None


def dump_process_info(process_info, output_path):
    """ serialize process_info back into trace file """
    output_trace = []
    for pid, pid_info in process_info.items():
        output_trace.append(
            {
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "name": "process_name",
                "args": {"name": pid_info["name"]},
            }
        )
        if "sort_index" in pid_info.keys():
            output_trace.append(
                {
                    "ph": "M",
                    "pid": pid,
                    "tid": 0,
                    "name": "process_sort_index",
                    "args": {"sort_index": pid_info["sort_index"]},
                }
            )

        for tid, tid_info in pid_info["threads"].items():
            output_trace.append(
                {
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "name": "thread_name",
                    "args": {"name": tid_info["name"]},
                }
            )
            if "sort_index" in tid_info.keys():
                output_trace.append(
                    {
                        "ph": "M",
                        "pid": pid,
                        "tid": tid,
                        "name": "thread_sort_index",
                        "args": {"sort_index": tid_info["sort_index"]},
                    }
                )

            for evt in tid_info["events"]:
                output_trace.append(
                    {
                        "ph": "X",
                        "pid": pid,
                        "tid": tid,
                        "name": evt["name"],
                        "ts": evt["ts"],
                        "dur": evt["dur"],
                        "args": evt["args"],
                    }
                )

    with open(output_path, "w", encoding='utf-8') as fp:
        json.dump(output_trace, fp)
