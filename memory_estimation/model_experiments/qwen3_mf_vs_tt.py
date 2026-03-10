from memory_estimation.estimate_v2 import EvaluatorV2


# e.set_strategy(full_rec=False, dp=2, mp=4)
# e.estimate_peak()
def make_dims(n):
    x = n
    res = []
    while x>=1:
        y = n // x
        res += [(x,y)]
        x //= 2
    return res

from memory_estimation.evaluators.tail import EvalTailSingle
from memory_estimation.evaluators.tail import EvalMTP
def activ_output(ccfg, ctx):
    # output softmax
    return 2*EvalTailSingle.activ_out_single(ccfg, ctx) + EvalMTP.activ_mtp(ccfg, ctx)
def custom(ccfg):
    # ccfg.emb_out_in_offset = False
    ccfg.bytes_grad = 4


e = EvaluatorV2("/mnt/nvme_1_3_4/philippe/gitcode/mindformers/configs/qwen3/pretrain_qwen3_1_7b_.yaml", log_level=0)
e.set_tail_eval_fun(dyn_activ=activ_output)

for r in [True, False]:
    print("FULL REC",r)
    for dp,tp in make_dims(8):
        e.set_strategy(full_rec=r,dp=dp, mp=tp)
        # e.print_ccfg()
        # e.print_ctx()
        e.estimate_peak()

e = EvaluatorV2("/mnt/nvme_1_3_4/philippe/torchtitan/torchtitan/models/qwen3/train_configs/qwen3_1.7b.toml", log_level=0)
e.set_tail_eval_fun(dyn_activ=activ_output)


for r in [True, False]:
    print("FULL REC",r)
    for dp,tp in make_dims(8):
        e.set_strategy(full_rec=r,dp=dp, mp=tp)
        e.set_ccfg(custom)
        # e.print_ccfg()
        # e.print_ctx()
        # print(e.get_strategy())
        e.estimate_peak()

# strats = [
#     # {"dp":8,"tp":1,"pp":1,"ep":8,"cp":1},
#     # {"dp":4,"tp":2,"pp":1,"ep":8,"cp":1},
#     # {"dp":2,"tp":4,"pp":1,"ep":8,"cp":1},
#     {"dp":4,"tp":1,"pp":2,"ep":4,"cp":1},
#     # {"dp":2,"tp":1,"pp":4,"ep":2,"cp":1},
#     # {"dp":2,"tp":2,"pp":1,"ep":8,"cp":2},
#     # {"dp":2,"tp":2,"pp":2,"ep":1,"cp":1}
# ]
# e = EvaluatorV2("/mnt/nvme_1_3_4/philippe/torchtitan/torchtitan/models/deepseek_v3/train_configs/deepseek_v3_0.9.toml", log_level=0)
# e.set_tail_eval_fun(dyn_activ=activ_output)
# for s in strats :
#     vp = 2 if s["pp"]>1 else 1
#     e.set_strategy(full_rec=False, sel_rec=False,dp=s["dp"], mp=s["tp"], pp=s["pp"], ep=s["ep"], cp=s["cp"], vpp=vp)
#     e.set_ccfg(custom)
#     e.estimate_peak(verbose=True)