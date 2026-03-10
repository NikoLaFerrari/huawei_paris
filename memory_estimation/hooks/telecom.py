from memory_estimation.hook_base import MemEvalHook, hook_runner
from paradise.common.arch_hooks import custom_deepseek3
from memory_estimation.evaluators.layer_block import EvalFFn
from memory_estimation.evaluators.tail import EvalTailSingle, EvalMTP
from memory_estimation.evaluators.body import EvalBody
from memory_estimation.evaluators.comm import EvalLayerComm
from memory_estimation.logger import logger
import math


class Telecom(MemEvalHook):
    def custom_lay_shard(ccfg):
        ccfg.shard_grad_exp = ccfg.ep
        # ccfg.shard_recompute_input = ccfg.t
        ccfg.is_shard_mtp_param = True
        ccfg.shard_output_activ = ccfg.t
        op_shard = math.gcd(
            (ccfg.d_exp * ccfg.ep) if ccfg.has_op else ccfg.ep, ccfg.n_exp
        )
        ccfg.shard_p_os_exp = op_shard * ccfg.cp * ccfg.t_exp

    def custom_ccfg(ccfg):
        custom_deepseek3(ccfg)
        # Wrap existing deepseek hooks with custom_shard
        ccfg.layer_custom_config_callback(Telecom.custom_lay_shard)

    def stat_embed_grad(ccfg, ctx):
        # Overwrite stat_embed by changing its grad sharding
        param_size = ctx.eval.num_p(ccfg, ctx)
        b_grad = ccfg.bytes_grad
        b_grad /= ccfg.cp * ccfg.t
        g = param_size * b_grad
        return g

    def stat_lay_muon_p(ccfg, ctx):
        if "muon" not in ccfg.optimizer.lower():
            return EvalBody.stat_p_layer(ccfg, ctx)
        non_exp_p, exp_p = ctx.eval.num_p(ccfg, ctx)
        p = exp_p / ccfg.shard_p_os_exp_partial
        p += non_exp_p / ccfg.shard_p_os_non_exp_partial
        return p * ccfg.bytes_p

    def stat_lay_muon_os(ccfg, ctx):
        if "muon" not in ccfg.optimizer.lower():
            return EvalBody.stat_os_layer(ccfg, ctx)
        # Only one optimizer state (muon_m)
        non_exp_p, exp_p = ctx.eval.num_p(ccfg, ctx)
        p = exp_p / ccfg.shard_p_os_exp_partial
        p += non_exp_p / ccfg.shard_p_os_non_exp_partial
        os = p * ccfg.bytes_os if not ctx.swap_os else 0
        return os

    def ffn_moe_activ(ccfg, ctx):
        micro_factor = ctx.micro_factor
        reshape = max(ccfg.n_ffMM, ccfg.n_ffBMM) * ccfg.hff * ccfg.h
        reshape *= ccfg.bytes_compute * ccfg.n_exp / ccfg.ep
        reshape *= micro_factor  # Not sharded
        return (
            EvalFFn.routed_exp_activations(ccfg, ctx)
            + reshape
            + EvalFFn.shared_exp_activations(ccfg, ctx)
            + EvalFFn.ffn_router_and_concat_activations(ccfg, ctx)
        )

    def dyn_fullrec_activ(ccfg, ctx):
        micro_factor = ctx.micro_factor
        forward_activation = (
            micro_factor * ccfg.bytes_compute * ccfg.s * ccfg.b * ccfg.h
        )
        forward_activation /= ccfg.shard_recompute_input
        # FA not recomputed, only one micro?
        ctx.micro_factor = 1
        attn_score = ctx.attn_score_activ(ccfg, ctx)
        ctx.micro_factor = micro_factor
        return forward_activation + attn_score

    def dyn_out_activ(ccfg, ctx):
        micro_factor = ctx.micro_factor
        # add oneHot activations from output softmax
        softmax_onehots = ccfg.s * ccfg.b * ccfg.v
        extra_activ = ccfg.bytes_softmax * softmax_onehots
        extra_activ_mtp = ccfg.n_mtp * ccfg.bytes_softmax * softmax_onehots
        extra_activ *= micro_factor
        extra_activ_mtp *= micro_factor
        return (
            sum(
                [
                    EvalTailSingle.activ_out_single(ccfg, ctx) + extra_activ,
                    EvalMTP.activ_mtp(ccfg, ctx) + extra_activ_mtp,
                ]
            )
            / ccfg.shard_output_activ
        )

    @hook_runner("deepseekv3")
    def run_hooks(e):
        e.set_ccfg(Telecom.custom_ccfg)
        e.set_head_eval_fun(stat_grad=Telecom.stat_embed_grad)
        e.set_ffn_eval_fun(moe_activ=Telecom.ffn_moe_activ)
        e.set_body_eval_fun(
            stat_p=Telecom.stat_lay_muon_p, stat_os=Telecom.stat_lay_muon_os
        )
        e.set_body_eval_fun(
            "FULL_REC_LAYER",
            dyn_activ=Telecom.dyn_fullrec_activ,
            dyn_tp_comm=EvalLayerComm.tp_comm_layer,
        )
        e.set_tail_eval_fun(dyn_activ=Telecom.dyn_out_activ)
        if e.get_strategy()["sched"] == "seqsmartvpp":
            e.set_passes(vpp_less_mem=True)