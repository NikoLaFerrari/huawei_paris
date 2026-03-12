from interface import Handler
import os

input_dims = {
        'dims': ['DP','MP','PP','MB','MBS'],
        'device_count': 16,
        'gbs': 64
        }

trace_path = [
        #"/home/pl/telecom/results_telecom_swap/bench_5c16339b00a3/output/profile/rank_0/localhost.localdomain_1024328_20251202125633690_ascend_ms/ASCEND_PROFILER_OUTPUT/trace_view.json",
        #"/home/pl/telecom/results_telecom_swap/bench_60fa0500cef8/output/profile/rank_0/localhost.localdomain_2502847_20251202152451316_ascend_ms/ASCEND_PROFILER_OUTPUT/trace_view.json",
        #"/home/pl/telecom/results_telecom_swap/bench_ab038b0026aa/output/profile/rank_0/localhost.localdomain_1369673_20251202133024019_ascend_ms/ASCEND_PROFILER_OUTPUT/trace_view.json",
        #"/home/pl/telecom/results_telecom_swap/bench_fd4e15cbdb5d/output/profile/rank_0/localhost.localdomain_2911547_20251204124021008_ascend_ms/ASCEND_PROFILER_OUTPUT/trace_view.json",
        #"/home/pl/telecom/results_telecom_swap/bench_e671bf09d8d0/output/profile/rank_0/localhost.localdomain_4160253_20251203141801909_ascend_ms/ASCEND_PROFILER_OUTPUT/trace_view.json",
        "./traces/trace3.json",
        "./traces/trace_view.json",
        "./traces/trace1.json",
        "./traces/trace2.json"
        ]

config_path = [
         #"/home/pl/telecom/results_telecom_swap/bench_5c16339b00a3/config_5c16339b00a3.yaml",
         #"/home/pl/telecom/results_telecom_swap/bench_60fa0500cef8/config_60fa0500cef8.yaml",
         #"/home/pl/telecom/results_telecom_swap/bench_ab038b0026aa/config_ab038b0026aa.yaml",
         #"/home/pl/telecom/results_telecom_swap/bench_fd4e15cbdb5d/config_fd4e15cbdb5d.yaml",
         #"/home/pl/telecom/results_telecom_swap/bench_e671bf09d8d0/config_e671bf09d8d0.yaml",
         "./configs/config_small.yaml",
         "./configs/config_small2.yaml",
         "./configs/config.yaml",
         "./configs/config.yaml"
         ]

graphs = [
         #"/home/pl/telecom/results_telecom_swap/bench_5c16339b00a3/graphs/",
         #"/home/pl/telecom/results_telecom_swap/bench_60fa0500cef8/graphs/",
         #"/home/pl/telecom/results_telecom_swap/bench_ab038b0026aa/graphs/",
         #"/home/pl/telecom/results_telecom_swap/bench_fd4e15cbdb5d/graphs/",
         #"/home/pl/telecom/results_telecom_swap/bench_e671bf09d8d0/graphs/",
         r"C:\Users\j50056410\Desktop\jagan\regression\graphs\graphs\graphs",
         r"C:\Users\j50056410\Desktop\jagan\regression\graphs\graphs\graphs\graphs",
         r"C:\Users\j50056410\Desktop\jagan\regression\graphs",
         r"C:\Users\j50056410\Desktop\jagan\regression\graphs\graphs"
        ]

input_config = "./configs/config_small2.yaml"#'/home/pl/telecom/results_telecom_swap/bench_5c16339b00a3/config_5c16339b00a3.yaml'

handler = Handler(trace_path, config_path, graphs, input_config, input_dims)

handler.run_calibration()
#print(f"[test_interface] r_out: {r_out}")
#print(f"thetas: {thetas}")
