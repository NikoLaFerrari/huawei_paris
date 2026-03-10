from memory_estimation.estimate_v2 import *
import time
import pprint

# e = EvaluatorV2(f"C:/Users/p84389243/AppData/Roaming/WeLink_Desktop/appdata/IM/p84389243/DownloadFiles/438b2.yaml", hook_cls=Telecom(), log_level=0)
# # e.estimate_peak(verbose=True)

# import cProfile


# def repeat():
#     times = []
#     start_global = time.time()
#     nb_it = 1000
#     for _ in range(nb_it):
#         start = time.time()
#         e.estimate_peak_insight()
#         end = time.time()
#         times += [end - start]
#         # print(f"Elapsed time: {end - start} seconds")
#     print(f"Num iteration: {nb_it}, Elapsed time: {time.time() - start_global} seconds")
#     print("min",min(times),"max",max(times),"med",times[nb_it//2],"mean",sum(times)/nb_it)

# cProfile.run('repeat()', sort='cumtime')

# # # ppb = e.estimate_layer_memory()
# # # pprint.pprint(ppb)
# insights = e.estimate_peak_insight()
# pprint.pprint(insights)
# stages=[s["Static"]+s["Dynamic"] for s in insights]
# pprint.pprint(stages)

from memory_estimation.score import mape, r2
test_m_peaks, real_m_peaks = [], []
n_test = 0

real = {
    "1119b": 55161,
    "1119b_norec": 136644,
    "112b": 18407,
    "112b_1024dp": 69776,
    "112b_256dp": 38951,
    "112b_norec": 59351,
    "438b2": 41973,
    "438b2_norec": 131568,
    "1119b_dp16tp4pp2ep16": 42013,
    "438b3": 45830,
    "438b3_r3": 38415,
    "438b3_r7_16K": 40464,
    "438b3_r11_16K": 28447,
    "438b3_r13": 42146,
    "105b_seq32k_norec": 162847,
    "105b_seq32k_norec_cp2": 126258,
    "105b_seq32k_norec_cp4": 88854,
    "105b_seq32k_fullrec": 65525,
    "105b_seq32k_fullrec_cp2": 69794,
    "105b_seq32k_fullrec_cp4": 52692
}


import os
from memory_estimation.hooks.telecom import Telecom
for _,_,files in os.walk("./telecom"):
    files+=["../../comparators/example/1119b_dp16tp4pp2ep16.yaml"]
    for f in sorted(files): 
        if "tiny" not in f and "105b" not in f and f.endswith("yaml"):
            n_test+=1
            e = EvaluatorV2(f"./telecom/{f}", log_level=0, hook_cls=Telecom())
            peak_mem = e.estimate_peak()
            e.mem_fit(peak_mem)
            insight = e.estimate_peak_insight()
            # e.all_stage_micro_factors()
            real_mem = real[f.split("/")[-1].split(".")[0]]
            real_m_peaks += [real_mem]
            test_m_peaks += [peak_mem]
            print("\n".join([str(i["Static"]+i["Dynamic"]) for i in insight]))
            print("peak acc", round(peak_mem/real_mem*100))

print("mape",mape(test_m_peaks, real_m_peaks))
print("r2",r2(test_m_peaks, real_m_peaks))
print("n_test", n_test)

# from memory_estimation.hooks.texthawk_hooks import XY
# e = EvaluatorV2("C:/Users/p84389243/toolkits/memory_estimation/test_cases/xy/vpp2.json", log_level=0, hook_cls=XY())
# e.estimate_peak(verbose=True)