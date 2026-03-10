import copy
import argparse
import pandas as pd
import csv
from functools import reduce
from bench_tools import prof
from bench_tools import ms_trace

class EventWaitAnalyzer:
    def tag_event(self, e, tag):
        return {"evt": e, "tag": tag}

    def tag_events(self, event_list, tag):
        tagged_list = []
        for e in event_list:
            tagged_list.append(self.tag_event(e, tag)) 
        return tagged_list       

    def compute_step_avg_causes(self, wait_causes, n_steps):
        ret = {}
        for w, e in wait_causes:
            tag = e["tag"] if e is not None else "unattributed"
            if tag not in ret.keys():
                ret[tag] = 0.
            ret[tag] += w["dur"]
        
        for k in ret.keys():
            ret[k] /= n_steps
        
        return ret

    def extract_swap_optimizer_events(self, evts):
        swap_events = []
        swap_optimizer_events = []
        for e in evts:
            if "mindspore_op" in e["args"].keys() and "optimizer-" in e["args"]["mindspore_op"]:
                swap_optimizer_events.append(e)
            else:
                swap_events.append(e)
        return swap_events, swap_optimizer_events

    def simple_communication_classification(self, communication_events):
        tagged_events = []
        collective_events = []
        p2p_events = []
        for e in communication_events:
            if e["name"].startswith("hcom_receive") or e["name"].startswith("hcom_send"):
                p2p_events.append(e)
            else:
                collective_events.append(e)
        tagged_events.extend(self.tag_events(collective_events, "collective"))
        tagged_events.extend(self.tag_events(p2p_events, "p2p"))
        return tagged_events
        
    def detailed_communication_classification(self, communication_events, comm_classification_res):
        tagged_events = []
        for e in communication_events:
            scope = e["args"]["mindspore_op"].removeprefix("Kernel::KernelLaunch::")
            tag = comm_classification_res[scope]
            tagged_events.append(self.tag_event(e, tag))
        return tagged_events

    def tag_communication_events(self, communication_events, comm_classification_res = None):    
        if comm_classification_res is not None:
            return self.detailed_communication_classification(communication_events, comm_classification_res)
        return self.simple_communication_classification(communication_events)

    def find_delaying_events(self, process_info, comm_classification_res = None):
        delaying_events = []

        # Communications
        communication_events = []
        communications_pid = ms_trace.find_communication_pid(process_info)
        for tid in prof.get_process_threads(process_info, communications_pid).keys():
            thread_communication_events = prof.get_thread_events(process_info, communications_pid, tid)
            thread_communication_events = [ e for e in thread_communication_events if e["name"].startswith("hcom_")]
            communication_events.extend(thread_communication_events)
        delaying_events.extend(self.tag_communication_events(communication_events, comm_classification_res))

        # SWAP
        compute_pid = ms_trace.find_compute_pid(process_info)
        swap_out_tid, swap_in_tid = ms_trace.find_swap_tids(process_info, compute_pid)
        optimizer_swap_events = []
        if swap_out_tid is not None:
            swap_out_events = prof.get_thread_events(process_info, compute_pid, swap_out_tid)
            swap_out_events = ms_trace.merge_swap_events(swap_out_events)
            swap_out_events, optimizer_swap_out_events = self.extract_swap_optimizer_events(swap_out_events)
            delaying_events.extend(self.tag_events(swap_out_events, "swap_out"))
            optimizer_swap_events.extend(optimizer_swap_out_events)
        
        if swap_in_tid is not None:
            swap_in_events = prof.get_thread_events(process_info, compute_pid, swap_in_tid)
            swap_in_events = ms_trace.merge_swap_events(swap_in_events)
            swap_in_events, optimizer_swap_in_events = self.extract_swap_optimizer_events(swap_in_events)
            delaying_events.extend(self.tag_events(swap_in_events, "swap_in"))
            optimizer_swap_events.extend(optimizer_swap_in_events)
        
        delaying_events.extend(self.tag_events(optimizer_swap_events, "optimizer_swap"))
        
        return delaying_events

    def get_compute_stream_events(self, process_info):
        compute_pid = ms_trace.find_compute_pid(process_info)
        kernels_tid = ms_trace.find_kernels_tid(process_info, compute_pid)
        return prof.get_thread_events(process_info, compute_pid, kernels_tid)

    def find_wait_events(self, evts):
        return [ e for e in evts if e["name"] == "EVENT_WAIT" ]

    def find_compute_wait_events(self, process_info):
        compute_pid = ms_trace.find_compute_pid(process_info)
        kernels_tid = ms_trace.find_kernels_tid(process_info, compute_pid)
        compute_wait_events = [ e for e in prof.get_thread_events(process_info, compute_pid, kernels_tid) if e["name"] == "EVENT_WAIT" ]
        
        return compute_wait_events

    def compute_total_compute_time(self, compute_events):
        total_compute = 0.
        for e in compute_events:
            if e["name"] in ["EVENT_WAIT", "EVENT_RECORD"]:
                continue
            total_compute += e["dur"]
        return total_compute

    def compute_total_idle_time(self, compute_events, step_events):
        total_idle = 0.
        last_event_end = step_events[0]["ts"]
        for e in compute_events:
            if e["name"] in ["EVENT_RECORD"]:
                continue
            total_idle += e["ts"] - last_event_end
            last_event_end = e["ts"] + e["dur"]
        return total_idle

    def add_trace_wait_causes(self, process_info, wait_causes, pid=None, tid=None):
        if pid is None or tid is None:
            pid = ms_trace.find_compute_pid(process_info)
            tid = ms_trace.find_kernels_tid(process_info, pid)
        for w, e in wait_causes:
            tag = e["tag"] if e is not None else "unattributed wait"
            prof.create_trace_event(process_info, pid, tid, f"{tag} wait", w["ts"], w["dur"], {})
    
    def dump_trace_wait_causes(self, process_info, out_path, wait_causes):
        new_process_info = copy.deepcopy(process_info)
        self.add_trace_wait_causes(new_process_info, wait_causes)
        prof.dump_process_info(new_process_info, out_path)

    def make_summary_df(self, wait_causes_totals):
        def add_data_pct(data, label, val, total=None):
            data["label"].append(label)
            data["time"].append(val)
            if total is not None:
                data["pct"].append((val / total) * 100)
            else:
                data["pct"].append("")
        step_time = wait_causes_totals["step_time"]
        total_wait = wait_causes_totals["total"]
        total_compute = wait_causes_totals["compute"]
        total_idle = wait_causes_totals["idle"]
        
        data = {
            "label": [],
            "time": [],
            "pct": [],
        }
        
        pd.options.display.float_format = "{:,.2f}".format
        
        add_data_pct(data, "step_time", step_time)
        add_data_pct(data, "compute", total_compute, step_time)
        add_data_pct(data, "idle", total_idle, step_time)
        add_data_pct(data, "total_wait", total_wait, step_time)
        
        ignore_keys = ["step_time", "total", "compute", "idle"]
        for tag, total in wait_causes_totals.items():
            if tag in ignore_keys:
                continue
            add_data_pct(data, f"{tag}_wait", total, step_time)
        data["time"] = [x / 1000 for x in data["time"]]
        df = pd.DataFrame.from_dict(data)
        df = df.sort_values(by=['time'], ascending=False)
        return df

    def find_wait_causes(self, wait_events, delaying_events, abs_tol=2000.):    
        # Sort delaying events by the time they end to limit the number of times they need to be looked
        delaying_events = sorted(delaying_events, key=lambda e: e["evt"]["ts"] + e["evt"]["dur"])
        wait_events = sorted(wait_events, key=lambda e: e["ts"] + e["dur"])    
        event_blacklist = ["EVENT_RECORD"]
        wait_causes = []
        #For each wait events, try to find a delaying event which ends before the wait event ends
        i_event = 0
        for w in wait_events:
            candidate_event = None
            w_end = w["ts"] + w["dur"]
            
            # Find the event which ends the closest before the wait ends
            while i_event < len(delaying_events):
                e = delaying_events[i_event]
                e_end = e["evt"]["ts"] + e["evt"]["dur"]
                end_diff = w_end - e_end            
                if end_diff < 0.:
                    break
                if end_diff < abs_tol and e["evt"]["name"] not in event_blacklist:
                    candidate_event = e
                i_event += 1
            
            wait_causes.append((w, candidate_event))
            candidate_event = None
        
        return wait_causes    

    def summarize_wait_causes(self, compute_events, wait_events, wait_causes, step_events):
        n_steps = len(step_events)
        step_time = reduce(lambda x, y: x + y["dur"], step_events, 0) / n_steps
        total_wait = reduce(lambda x, y: x + y["dur"], wait_events, 0) / n_steps
        
        wait_causes_totals = self.compute_step_avg_causes(wait_causes, n_steps)
        wait_causes_totals["compute"] = self.compute_total_compute_time(compute_events) / n_steps
        wait_causes_totals["idle"] = self.compute_total_idle_time(compute_events, step_events) / n_steps
        wait_causes_totals["step_time"] = step_time
        wait_causes_totals["total"] = total_wait

        return wait_causes_totals

    def print_wait_cause_summary(self, summary):
        df = self.make_summary_df(summary)
        print(df.to_string(index=False))
    
    def dump_summary_csv(self, summary, path):
        df = self.make_summary_df(summary)
        df.to_csv(path, sep=',', encoding='utf-8', index=False, header=True)
    
    def dump_wait_cause_trace(self, process_info, wait_causes, path):
        new_process_info = copy.deepcopy(process_info)
        pid = ms_trace.find_compute_pid(new_process_info)
        tid = ms_trace.find_kernels_tid(new_process_info, pid)
        self.add_trace_wait_causes(new_process_info, wait_causes, pid, tid)

        kernel_events = prof.get_thread_events(new_process_info, pid, tid)
        kernel_events = [ e for e in kernel_events if e["name"] not in ["EVENT_WAIT", "EVENT_RECORD"] and e["dur"] > 1 ]
        kernel_events = sorted(kernel_events, key=lambda d: d["ts"])
        prof.set_thread_events(new_process_info, pid, tid, kernel_events)

        prof.dump_process_info(new_process_info, path)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('trace', help='Input trace path')
    parser.add_argument('-s','--summary-csv', help='path for dumping to output csv')
    parser.add_argument('-t','--dump-trace', help='path for dumping to output csv')
    args = parser.parse_args()

    trace_path = args.trace
    csv_path = args.summary_csv
    dump_path_trace = args.dump_trace

    process_info = prof.parse_process_info(trace_path)

    analyzer = EventWaitAnalyzer()

    compute_events = sorted(analyzer.get_compute_stream_events(process_info), key=lambda d: d["ts"])
    wait_events = analyzer.find_wait_events(compute_events)
    delaying_events = analyzer.find_delaying_events(process_info)
    wait_causes = analyzer.find_wait_causes(wait_events, delaying_events)

    step_events = sorted(ms_trace.find_step_events(process_info) , key=lambda d: d["ts"])
    summary = analyzer.summarize_wait_causes(compute_events, wait_events, wait_causes, step_events)

    analyzer.print_wait_cause_summary(summary)

    if csv_path is not None:
        print("dumping summary info to", csv_path)
        analyzer.dump_summary_csv(summary, csv_path)
    
    if dump_path_trace is not None:
        print("dumping detailed info on trace", dump_path_trace)
        analyzer.dump_wait_cause_trace(process_info, wait_causes, dump_path_trace)

if __name__ == "__main__":
    main()
