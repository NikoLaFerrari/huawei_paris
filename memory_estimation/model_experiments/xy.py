from memory_estimation.estimate_v2 import EvaluatorV2
import pprint
e = EvaluatorV2("/mnt/nvme_1_3_4/philippe/toolkits/memory_estimation/model_experiments/hq_half_pp.json", log_level=0, hook_cls="XY")
pprint.pprint(e.get_strategy())
e.estimate_peak()

e.set_strategy(model_name="deepseekv3",dp=4,mp=16)
pprint.pprint(e.get_strategy())
print(e.mem_fit(e.estimate_peak()))