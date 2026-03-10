from memory_estimation.estimate_v2 import *
from memory_estimation.hooks.texthawk_hooks import XY
from pprint import pprint

# Instantiate evaluator with a model configuration, log_level=0 removes warning messages
# 8p
# e = EvaluatorV2("/mnt/nvme_1_3_4/philippe/toolkits/memory_estimation/model_experiments/xy_test1.json",log_level=0)
# e = EvaluatorV2("/mnt/nvme_1_3_4/philippe/toolkits/memory_estimation/model_experiments/xy_test1_vpp2.json",log_level=0)
# e = EvaluatorV2("/mnt/nvme_1_3_4/philippe/toolkits/memory_estimation/model_experiments/xy_test1_fullrec.json",log_level=0)
# e = EvaluatorV2("/mnt/nvme_1_3_4/philippe/model_pretrain_8p_xunyi.json", log_level=0)
# 128p
# e = EvaluatorV2("/mnt/nvme_1_3_4/philippe/new_XY/MindSpeed-MM-XY/examples/mm_model/texthawk_ds/8p_dryrun/model_norec.json",log_level=0)
# 512p
# e = EvaluatorV2("/mnt/nvme_1_3_4/philippe/toolkits/memory_estimation/model_experiments/vpp2.json",log_level=0)
# e = EvaluatorV2("/mnt/nvme_1_3_4/philippe/toolkits/memory_estimation/model_experiments/vpp2_tt.json", log_level=0)
# e = EvaluatorV2(
#     "./vpp2_18_ViT.json",
#     log_level=0,
#     hook_cls=XY(),
# )
# # pprint(e.get_strategy())
# e.estimate_peak()
# e.set_strategy(model_name="deepseekv3", dp=4, tp=8, full_rec=False)
# pprint(e.get_strategy())
# e.estimate_peak()
# e.print_ccfg()
# e.print_ctx()
# print(vars(e))
# print(dir(e))
# pprint.pprint(e.logs_mem_stage(0))

# e = EvaluatorV2("/mnt/nvme_1_3_4/philippe/toolkits/memory_estimation/model_experiments/hq.json",log_level=0, hook_cls=XY())
# # e.estimate_peak(verbose=True)
# ppb = e.estimate_layer_memory()
# pprint.pprint(ppb)
e = EvaluatorV2("hq_half_pp.json",log_level=0, hook_cls=XY())
e.estimate_peak(verbose=True)
ppb = e.estimate_layer_memory()
pprint(ppb)
# e.all_stages_micro_factors()
