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

"""identifies comm, comp events and their TID, PID"""

from math import isclose

from bench_tools import prof


def find_compute_pid(process_info):
    """ return pid of Ascend Hardware process in the trace """
    return prof.find_process_id(process_info, "Ascend Hardware")


def find_communication_pid(process_info):
    """ return pid of the comm process in the trace """
    return prof.find_process_id(process_info, "Communication")


def find_ms_pid(process_info):
    """ return teh pid of the MindSpore process in the trace """
    return prof.find_process_id(process_info, "MindSpore")


def find_scope_pid(process_info):
    """ return the pid of the 'Scope Layer' process in the trace"""
    return prof.find_process_id(process_info, "Scope Layer")


def get_sorted_tids(process_info, pid):
    """ return the sorted thread ids for a given pid"""
    return sorted(prof.get_process_threads(process_info, pid).keys())


def find_kernels_tid(process_info, compute_pid):
    """ 
    heuristically find teh comp thread (tid)
    that contains kernel events 
    """
    tids = get_sorted_tids(process_info, compute_pid)
    for tid in tids:
        evts = prof.get_thread_events(process_info, compute_pid, tid)
        if len(evts) > 1:
            return tid
    return None


def find_communication_events(process_info, pid):
    """ collect all HCCL comm events for a pid """
    communication_events = []
    for tid in prof.get_process_threads(process_info, pid).keys():
        thread_communication_events = prof.get_thread_events(
            process_info, pid, tid
        )
        thread_communication_events = [
            e
            for e in thread_communication_events
            if e["name"].startswith("hcom_")
        ]
        communication_events.extend(thread_communication_events)
    return communication_events


def find_swap_tids(process_info, pid):
    """ detect two MEMCPY_ASYNC threads used for swap-out/swap-in """
    copy_threads = []
    pid_info = process_info[pid]
    for tid, tid_info in pid_info["threads"].items():
        n_events = len(tid_info["events"])
        if n_events == 0:
            continue
        n_copy = 0
        for e in tid_info["events"]:
            if (
                e["name"] == "MEMCPY_ASYNC"
                and "mindspore_op" in e["args"].keys()
                and "MoveTo-op" in e["args"]["mindspore_op"]
            ):
                n_copy += 1
        if n_copy > 0:
            copy_threads.append({"tid": tid, "n_copy": n_copy})
    if len(copy_threads) < 2:
        return None, None

    copy_threads = sorted(
        copy_threads, key=lambda d: d["n_copy"], reverse=True
    )
    copy_threads = copy_threads[0:2]
    if not isclose(
        copy_threads[0]["n_copy"], copy_threads[1]["n_copy"], rel_tol=0.40
    ):
        return None, None

    copy_threads = [d["tid"] for d in copy_threads]
    copy_threads = sorted(copy_threads)

    swap_out_tid, swap_in_tid = copy_threads

    return swap_out_tid, swap_in_tid


def merge_swap_events(events):
    """ 
    merge MEMCPY_ASYNC swap events by 
    connection_id into coarse MoveTo events 
    """
    connection_events = {}
    for move_op in events:
        if move_op["name"] != "MEMCPY_ASYNC":
            continue
        connection_id = move_op["args"]["connection_id"]
        if connection_id not in connection_events.keys():
            connection_events[connection_id] = {
                "name": "MoveTo",
                "ts": move_op["ts"],
                "dur": move_op["dur"],
            }
        else:
            connection_events[connection_id]["ts"] = min(
                connection_events[connection_id]["ts"], move_op["ts"]
            )
            connection_events[connection_id]["dur"] += move_op["dur"]
        if "mindspore_op" in move_op["args"].keys():
            connection_events[connection_id]["args"] = move_op["args"]
    return sorted(connection_events.values(), key=lambda d: d["ts"])


def find_step_events(process_info, step_key="RunGraph"):
    """ 
    return MindSpore step events whose name contains 'step_key'
    from the first MindSpore thread 
    """
    ms_pid = find_ms_pid(process_info)
    sorted_tids = get_sorted_tids(process_info, ms_pid)
    step_tid = sorted_tids[0]
    step_events = [
        e
        for e in prof.get_thread_events(process_info, ms_pid, step_tid)
        if step_key in e["name"]
    ]

    return step_events


def parse_layer_id(name):
    """ 
    parse the integer layer id prefix 
    from strings like '<layer>-<rest>'
    """
    sep_idx = name.index("-")
    layer_id = int(name[:sep_idx])
    return layer_id
