# pylint: skip-file
from memory_estimation.hook_base import MemEvalHook, hook_runner
from paradise.common.arch_hooks import custom_deepseek3, custom_default_transformer
import math


class ViT(MemEvalHook):
    """Vision Encoder hooks"""

    def custom_ccfg_vit(ccfg):
        custom_default_transformer(ccfg)
        ccfg.patch_size = ccfg.config.image_encoder.vision_encoder.patch_size
        ccfg.img_size = ccfg.config.image_encoder.vision_encoder.image_size
        ccfg.class_tok_len = (
            ccfg.config.image_encoder.vision_encoder.class_token_len
            if ccfg.config.image_encoder.vision_encoder.add_class_token
            else 0
        )
        ccfg.s = (ccfg.img_size // ccfg.patch_size) ** 2 + ccfg.class_tok_len
        ccfg.channels = 3  # RGB, maybe present in yaml
        ccfg.h_flatten = ccfg.channels * ccfg.patch_size**2
        ccfg.v = 1
        ccfg.b = 97
        ccfg.s_fa = ccfg.s / ccfg.a
        ccfg.shard_recompute_input = ccfg.t

    def n_param_patch_emb(ccfg, ctx):
        pos_emb = ccfg.h * ccfg.s  # ccfg.class_tok_len * ccfg.h
        conv2d = ccfg.h * ccfg.h_flatten  # same as patch**2*channel*h
        conv2d_bias = ccfg.h
        return pos_emb + conv2d + conv2d_bias

    def activ_patch_emb(ccfg, ctx):
        micro_factor = ctx.micro_factor
        grid_size = math.floor(
            (ccfg.img_size - (ccfg.patch_size - 1) - 1) / ccfg.patch_size + 1
        )
        activ_size = (
            micro_factor * ccfg.bytes_compute * ccfg.s * ccfg.b * ccfg.h
            + grid_size**2 * ccfg.b * ccfg.h
        )
        return activ_size / (ccfg.t * ccfg.cp)

    def n_param_vit_output(ccfg, ctx):
        # return ccfg.h * ccfg.h + ccfg.h * ccfg.n_class + ccfg.n_class
        return 2 * ccfg.h  # final norm

    def activ_vit_output(ccfg, ctx):
        last_norm = ccfg.s * ccfg.b * ccfg.bytes_norm * ccfg.h
        micro_factor = ctx.micro_factor
        activ_size = micro_factor * last_norm
        return activ_size / ccfg.t

    @hook_runner("siglip")
    def run_hooks(evaluator):
        Cls = ViT
        evaluator.set_ccfg(Cls.custom_ccfg_vit)
        evaluator.set_head_eval_fun(
            num_p=Cls.n_param_patch_emb, dyn_activ=Cls.activ_patch_emb
        )
        evaluator.set_tail_eval_fun(
            num_p=Cls.n_param_vit_output,
            dyn_activ=Cls.activ_vit_output,
            dyn_comm=0
        )

    def custom_ccfg_vit_qwen(ccfg):
        ccfg.img_size = 28
        ccfg.patch_size = ccfg.config.image_encoder.vision_encoder.patch_size
        ccfg.s = (ccfg.img_size // ccfg.patch_size) ** 2 + ccfg.class_tok_len
        ccfg.b = 97
        ccfg.s_fa = ccfg.s / ccfg.a

    @hook_runner("qwen2vit")
    def run_hooks_2(evaluator):
        ViT.run_hooks(evaluator)
        evaluator.set_ccfg(ViT.custom_ccfg_vit_qwen)
        evaluator.set_strategy(mp=1)

class Downsampler(MemEvalHook):
    """Downsampler hooks"""

    # SPE
    def custom_ccfg_ppn(ccfg):
        ccfg.s = ccfg.config.image_encoder.vision_encoder.spe_seq_len  # ViT
        ccfg.h = ccfg.config.image_encoder.vision_encoder.hidden_size  # ViT
        ccfg.b = 97  # MAX_NUM_IMG
        ccfg.full_rec = True
        ccfg.shard_recompute_input = ccfg.t

    def n_param_spe(ccfg, ctx):
        h = ccfg.config.image_encoder.vision_encoder.hidden_size  # ViT
        a = ccfg.config.image_encoder.vision_encoder.num_attention_heads  # ViT
        return 4 * h + h / a + a  # 2 * (2 * h/a * a) = 4h

    def activ_spe(ccfg, ctx):
        return ccfg.bytes_compute * ccfg.s * ccfg.b * ccfg.h

    # PPN
    def n_param_ppn(ccfg, ctx):
        return 2 * ccfg.h * ccfg.hff + Downsampler.n_param_spe(ccfg, ctx), 0

    def activ_ppn(ccfg, ctx):  #def dyn_ppn(ccfg, ctx):
        grid_size = math.floor((math.sqrt(ccfg.s) - 2) / 2 + 1)
        activ = ccfg.bytes_compute * (
            ccfg.s * ccfg.b * ccfg.hff
            + grid_size**2 * ccfg.b * ccfg.hff
            + ccfg.s * ccfg.b * ccfg.h
        )
        activ += Downsampler.activ_spe(ccfg, ctx)
        micro_factor = ctx.micro_factor
        if ctx.ppb:
            micro_factor = 1
        return micro_factor * activ / ccfg.t

    def fullrec_activ_ppn(ccfg, ctx):
        micro_factor = ctx.micro_factor
        activ = micro_factor * Downsampler.activ_spe(ccfg, ctx)
        activ /= ccfg.shard_recompute_input
        return activ  #, 0

    @hook_runner("ppn")
    def run_hooks(evaluator):
        Cls = Downsampler
        evaluator.set_ccfg(Cls.custom_ccfg_ppn)
        evaluator.set_head_eval_fun(0)
        evaluator.set_body_eval_fun(
            lay_type="NOT_REC_LAYER", num_p=Cls.n_param_ppn,
            dyn_activ=Cls.activ_ppn,
            dyn_comm=0
        )
        evaluator.set_body_eval_fun(
            lay_type="FULL_REC_LAYER",
            num_p=Cls.n_param_ppn,
            dyn_activ=Cls.fullrec_activ_ppn,
            dyn_comm=0
        )
        evaluator.set_body_eval_fun(
            lay_type="SEL_REC_LAYER", num_p=Cls.n_param_ppn,
            dyn_activ=Cls.activ_ppn,
            dyn_comm=0
        )
        evaluator.set_tail_eval_fun(0)


class Resampler(MemEvalHook):
    # QFormer
    def custom_ccfg_qformer(ccfg):
        custom_default_transformer(ccfg)
        ccfg.patch_size = ccfg.config.image_encoder.vision_encoder.patch_size
        ccfg.img_size = ccfg.config.image_encoder.vision_encoder.image_size
        ccfg.class_tok_len = (
            ccfg.config.image_encoder.vision_encoder.class_token_len
            if ccfg.config.image_encoder.vision_encoder.add_class_token
            else 0
        )
        ccfg.s = (ccfg.img_size // ccfg.patch_size) ** 2 + ccfg.class_tok_len
        ccfg.s_ratio = (
            ccfg.config.image_encoder.vision_projector.resampler.seq_ratio
        )
        ccfg.s /= ccfg.s_ratio
        ccfg.s_fa = ccfg.s / ccfg.a
        # ccfg.s = ccfg.config.qformer_seq_len
        ccfg.h_llm = (
            ccfg.config.image_encoder.vision_projector.resampler.llm_hidden_size
        )
        ccfg.v = 1
        ccfg.b = 97
        # Cross attention
        ccfg.n_attMM *= 2  # Later: Maybe 5
        ccfg.n_attBMM *= 2  # Later: Maybe 3
        ccfg.n_attParamCast = ccfg.n_attMM if not ccfg.has_op else 0
        ccfg.n_softmax += 1
        ccfg.n_normOp += 1
        ccfg.n_gather += 2
        # Update num layer
        ccfg.n_lay = len(
            ccfg.config.image_encoder.vision_encoder.selected_layers
        )
        img_decode_pp_partition = (
            ccfg.config.image_encoder.vision_encoder.pipeline_num_layers
        )
        # Try to insert all layer after ViT, following ViT pp partition
        stage_insert_idx, chunk_insert_idx = 0, 0
        num_layer_per_stage = max(1, ccfg.n_lay // ccfg.p // ccfg.vp)
        ccfg.offset = [
            [-num_layer_per_stage for _ in range(ccfg.p)]  # empty stages
            for _ in range(ccfg.vp)
        ]
        if img_decode_pp_partition:
            if isinstance(img_decode_pp_partition[0], list):  # vpp
                put = False
                for v_idx in range(ccfg.vp - 1, -1, -1):
                    for s_idx in range(ccfg.p - 1, -1, -1):
                        if img_decode_pp_partition[v_idx][s_idx]:
                            stage_insert_idx = s_idx
                            chunk_insert_idx = v_idx
                            put = True
                            break
                    if put:
                        break
            else:
                stage_insert_idx = ccfg.p - 1
                for p in img_decode_pp_partition[::-1]:
                    if p:
                        break
                    else:
                        stage_insert_idx = max(0, stage_insert_idx - 1)
        ccfg.offset[chunk_insert_idx][stage_insert_idx] = (
            ccfg.n_lay - num_layer_per_stage
        )
        ccfg.full_rec = True
        ccfg.sel_rec = False

    def n_param_qformer_output(ccfg, ctx):
        # with Dense layer
        dense = ccfg.s_ratio * ccfg.h * ccfg.h_llm
        return dense

    def activ_qformer_output(ccfg, ctx):
        # with Dense layer
        last_norm = ccfg.s * ccfg.b * ccfg.bytes_norm * ccfg.h
        dense = ccfg.bytes_compute * ccfg.s * ccfg.b * ccfg.h_llm
        activ_size = last_norm + dense
        micro_factor = ctx.micro_factor
        return micro_factor * activ_size / ccfg.t

    def comm_qformer_output(ccfg, ctx):
        dp_comm_size = (
            ccfg.comm_d_non_exp
            * ctx.eval.num_p(ccfg, ctx)
            / (ccfg.t * ccfg.cp)
        )
        return dp_comm_size

    @hook_runner("qformer")
    def run_hooks(evaluator):
        Cls = Resampler
        evaluator.set_ccfg(Cls.custom_ccfg_qformer)
        evaluator.set_head_eval_fun(0)
        evaluator.set_tail_eval_fun(
            num_p=Cls.n_param_qformer_output,
            dyn_activ=Cls.activ_qformer_output,
            dyn_dp_comm=Cls.comm_qformer_output
        )

class Projector(MemEvalHook):
    def custom_ccfg_proj(ccfg):
        #Remove all attention related variables
        ccfg.n_attMM = 0
        ccfg.n_attBMM = 0
        ccfg.n_attParamCast = 0
        ccfg.n_softmax = 0
        ccfg.n_normOp -= 1
        ccfg.h = ccfg.config.image_encoder.vision_projector.input_size

    @hook_runner("lnmlp")
    def run_hooks(evaluator):
        # MLP + Norm module
        evaluator.set_ccfg(Projector.custom_ccfg_proj)
        evaluator.set_head_eval_fun(0)
        evaluator.set_tail_eval_fun(0)
        evaluator.set_strategy(mp=1)

class LLM(MemEvalHook):
    """LLM hooks"""

    def custom_llm(ccfg):
        custom_deepseek3(ccfg)

    @hook_runner("deepseekv3")
    def run_hooks(evaluator):
        evaluator.set_ccfg(LLM.custom_llm)


class XY(ViT, Downsampler, Resampler, LLM):
    pass


class XY_Qwen(ViT, Projector, LLM):
    pass