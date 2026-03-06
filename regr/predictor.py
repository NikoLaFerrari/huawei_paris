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
"""
Predictor: extrapolates per-primitive Hockney formulae to a target
parallel configuration.

Hockney model
─────────────
For a single primitive invocation:

    T(n) = α + n / β

    α  [µs]               startup latency
    β  [unit / µs]        effective bandwidth (unit = bytes or FLOPs)
    n  [bytes | FLOPs]    work metric — see "variable" field in formula

The total predicted time for a primitive over one training step is:

    T_primitive = count_per_step × (α_eff + n_eff / β)

where α_eff and n_eff are the latency and effective work metric after
applying config-driven scaling (see _compute_scales).

Group-size-aware alpha scaling
──────────────────────────────
For ring-based collectives (DP_COMM, MP_COMM) the startup latency α
scales with the number of ring hops (p − 1).  Previously the predictor
inferred the base group size from config dims (e.g. mp_base), but the
Interpreter now records the actual measured group size from group_rank_ids
in each formula entry.

We therefore use formula["group_size"] as p_base for ring_hop_scale
wherever it is available (> 0), falling back to the config-derived degree
only when the IR lookup failed.

    alpha_eff = α × ring_hop_scale(p_base, p_target)

FLOPs vs bytes n_scale for COMPUTE
────────────────────────────────────
When the Interpreter fitted a MatMul-family primitive using FLOPs as x
(formula["variable"] == "flops"), the work metric scales as:

    forward / backward:  n_scale = mb_target / mb_base
        FLOPs ∝ batch size for dense matmuls.

    recompute:           n_scale = (pp_target / pp_base) × (mb_target / mb_base)
        Recompute is triggered at pipeline stage boundaries.  The number of
        recompute events per step scales with the number of pipeline stages
        that must re-run their forward pass during the backward phase.

For bytes-based compute the same scaling applies since output volume also
grows linearly with microbatch size.

Overlap model (step-level combination)
───────────────────────────────────────
    T_total = T_execution + T_blocking + T_async

    T_execution = (T_COMPUTE + T_BUBBLE) / vpp
        Virtual pipeline interleaving cuts effective compute+bubble time.

    T_blocking = T_MP_COMM × coeff_mp
        Tensor-parallel AllReduce / AllGather / ReduceScatter sits on the
        critical path (between forward and backward sub-layers).
        coeff_mp is the non-overlapped fraction.

    T_async = max(T_DP_COMM × coeff_dp, T_EP_COMM × coeff_ep)
        Gradient AR (DP) and MoE expert dispatch (EP) overlap with the
        backward/forward pass.  The non-overlapped tail is the bottleneck.
        We take max because both share the same NIC/HBM bandwidth.

    T_PP_COMM is NOT added here because pipeline communication time is
    already embedded in T_BUBBLE (the downstream rank is idle while
    waiting for the upstream activation).  Adding it again would double-
    count.  PP_COMM is tracked separately in bucket_times for analysis.

coeff_* ∈ [0, 1] are calibrated by Handler.calibrate_coefficients().

UNKNOWN_COMM lane
─────────────────
Primitives that the CommunicationClassifier could not match to a known
IR scope are placed in UNKNOWN_COMM.  Their total time is reported in the
breakdown but not added to the step time prediction, since we cannot
determine their overlap properties.  A non-zero UNKNOWN_COMM total is a
signal that the IR graph / trace alignment is incomplete.
"""

from collections import defaultdict

from paradise.dimensions import ALL_DIMS as AD

class Predictor:
    """Predicts step execution time for a target parallel configuration."""

    _DEFAULT_COEFFS = {"mp": 0.95, "dp": 0.10, "ep": 0.40}

    def __init__(self, model, base_dims, coeffs=None):
        """
        Parameters
        ----------
        model : dict[str, dict]
            Per-partition Hockney parameters from Interpreter.run_interpreter().
            Keys are sub-partitioned: "lane::primitive[dtype|alg_type]"
            or "lane::primitive[pass_type]".

            Required fields per entry:
                alpha, beta, n_mean, count_per_step, variable,
                group_size, low_confidence, pass_type

        base_dims : dict
            Parallel dimensions of the training traces used for regression.
            Expected keys: dp, mp, pp, ep, mb, vpp.

        coeffs : dict, optional
            Non-overlapped fractions {"mp": float, "dp": float, "ep": float}.
            Defaults to _DEFAULT_COEFFS if None.
        """
        self.model     = model
        self.base_dims = base_dims
        self.coeffs    = coeffs if coeffs is not None else dict(self._DEFAULT_COEFFS)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def predict(self, pred_dims):
        """Return predicted total step time (µs) for pred_dims."""
        total, _ = self.predict_with_breakdown(pred_dims)
        return total

    def predict_with_breakdown(self, pred_dims):
        """
        Predict step time with a full per-lane and per-primitive breakdown.

        Parameters
        ----------
        pred_dims : dict
            Target parallel configuration (dp, mp, pp, ep, mb, vpp).

        Returns
        -------
        total_time : float
            Predicted step duration, µs.

        breakdown : dict
            {
                "bucket_times": {lane: float},          # µs per lane
                "primitive_times": {part_key: float},   # µs per partition
                "low_confidence_lanes": {lane: [reason_str, ...]},
                "unknown_comm_µs": float,               # not in total_time
                "overlap_terms": {                      # intermediate values
                    "execution": float,
                    "blocking":  float,
                    "async":     float,
                },
            }
        """
        if 'VPP' in pred_dims.keys():
            vpp = max(1, pred_dims.get("vpp", 1)) 
        else:
            vpp = 1

        known_lanes = {
            "COMPUTE", "BUBBLE",
            "DP_COMM", "MP_COMM", "EP_COMM", "PP_COMM",
            "UNKNOWN_COMM",
        }
        bucket_times    = {lane: 0.0 for lane in known_lanes}
        primitive_times = {}
        low_conf_lanes  = defaultdict(list)

        for part_key, f in self.model.items():
            lane           = part_key.split("::")[0]
            primitive_time = self._predict_primitive(part_key, f, pred_dims)
            primitive_times[part_key] = primitive_time

            if lane not in bucket_times:
                bucket_times[lane] = 0.0
            bucket_times[lane] += primitive_time

            # Track low-confidence contributions per lane
            if f.get("low_confidence", False) and primitive_time > 0:
                reason = f.get("confidence_reason", "unknown reason")
                low_conf_lanes[lane].append(
                    f"{part_key.split('::', 1)[1]}: {reason}"
                )

        # ── Overlap model ─────────────────────────────────────────────
        compute = bucket_times.get("COMPUTE",  0.0)
        bubble  = bucket_times.get("BUBBLE",   0.0)
        mp_comm = bucket_times.get("MP_COMM",  0.0)
        dp_comm = bucket_times.get("DP_COMM",  0.0)
        ep_comm = bucket_times.get("EP_COMM",  0.0)
        # PP_COMM is tracked but not added (embedded in BUBBLE; see module doc)
        # UNKNOWN_COMM is tracked but not added (overlap properties unknown)

        execution = (compute + bubble) / vpp
        blocking  = mp_comm * self.coeffs["mp"]
        async_net = max(
            dp_comm * self.coeffs["dp"],
            ep_comm * self.coeffs["ep"],
        )

        total_time = execution + blocking + async_net

        breakdown = {
            "bucket_times":         bucket_times,
            "primitive_times":      primitive_times,
            "low_confidence_lanes": dict(low_conf_lanes),
            "unknown_comm_µs":      bucket_times.get("UNKNOWN_COMM", 0.0),
            "overlap_terms": {
                "execution": execution,
                "blocking":  blocking,
                "async":     async_net,
            },
        }

        self._print_breakdown(pred_dims, total_time, breakdown)
        return float(total_time), breakdown

    # ------------------------------------------------------------------ #
    #  Per-primitive prediction                                            #
    # ------------------------------------------------------------------ #

    def _predict_primitive(self, part_key, f, pred_dims):
        """
        Predict total time contribution of one partition for one step.

            T = count_per_step × (α_eff + n_eff / β)

        Parameters
        ----------
        part_key  : str    sub-partitioned key
        f         : dict   Hockney formula entry
        pred_dims : dict   target parallel configuration

        Returns
        -------
        float  predicted time contribution, µs  (≥ 0)
        """
        alpha          = f["alpha"]
        beta           = f["beta"]
        n_mean         = f["n_mean"]
        count_per_step = f["count_per_step"]
        variable       = f.get("variable", "bytes")
        pass_type      = f.get("pass_type", "unknown")

        lane = part_key.split("::")[0]
        alpha_scale, n_scale = self._compute_scales(
            lane, pred_dims, f, pass_type, variable
        )

        if lane == 'MP_COMM':
            check = pred_dims.get('mp', pred_dims.get('MP', self.base_dims.get('mp',1)))
            if check <= 1: return 0.0
        if lane == "DP_COMM":
            check = pred_dims.get("dp", pred_dims.get("DP", self.base_dims.get('dp',1)))
            if check <= 1: return 0.0
        if lane == "EP_COMM":
            check = pred_dims.get("ep", pred_dims.get("EP", 1))
            if check <= 1: return 0.0

        alpha_eff = alpha * alpha_scale
        n_eff     = n_mean * n_scale

        if beta == float("inf") or beta <= 0.0:
            time_per_call = alpha_eff
        else:
            time_per_call = alpha_eff + (n_eff / beta)

        return max(0.0, time_per_call * count_per_step)

    # ------------------------------------------------------------------ #
    #  Scaling rules                                                       #
    # ------------------------------------------------------------------ #

    def _compute_scales(self, lane, pred_dims, f, pass_type, variable):
        """
        Return (alpha_scale, n_scale) for a primitive.

        Alpha scaling uses the formula's measured group_size as p_base
        where available (> 0), falling back to config-derived degree.

        Scaling physics per lane
        ─────────────────────────

        DP_COMM  — data-parallel gradient AllReduce (ring)
            α_scale : ring hops grow with (dp − 1).
                      p_base = formula group_size if available, else dp_base.
                      p_target = dp_target.
            n_scale : gradient shard ∝ 1/mp (TP reduces per-rank param slice).
                      n_scale = mp_base / mp_target.

        MP_COMM  — model/tensor-parallel AllReduce / AllGather / RS (ring)
            α_scale : ring hops grow with (mp − 1).
                      p_base = formula group_size if available, else mp_base.
                      p_target = mp_target.
            n_scale : activation slice ∝ 1/mp.
                      n_scale = mp_base / mp_target.

        EP_COMM  — expert-parallel AllToAll (single-round exchange)
            α_scale : 1.0 — AllToAll latency does not grow with degree.
            n_scale : token volume ∝ 1/(ep × dp).
                      n_scale = (ep_base × dp_base) / (ep_target × dp_target).

        PP_COMM  — pipeline point-to-point Send/Receive
            α_scale : 1.0 — activation tensor size independent of pp degree.
            n_scale : 1.0 — same argument.

        COMPUTE (forward / backward)
            α_scale : 1.0.
            n_scale :
              bytes  → mb_target / mb_base  (output volume ∝ batch size)
              flops  → mb_target / mb_base  (FLOPs ∝ batch size for matmul)

        COMPUTE (recompute)
            α_scale : 1.0.
            n_scale :
              The number of recompute events per step grows with pp degree
              (more pipeline stages → more activation checkpoints to replay).
              n_scale = (pp_target / pp_base) × (mb_target / mb_base).

        BUBBLE
            α_scale : idle time grows with pipeline warmup/drain stages.
                      alpha_scale = (pp_target − 1) / max(1, pp_base − 1).
            n_scale : 1.0 (β = ∞, n_scale is unused).

        UNKNOWN_COMM
            1.0, 1.0 — overlap unknown; pass-through.

        Parameters
        ----------
        lane      : str
        pred_dims : dict   target configuration
        f         : dict   formula entry (for group_size)
        pass_type : str    "forward" | "backward" | "recompute" | "unknown"
        variable  : str    "bytes" | "flops"

        Returns
        -------
        alpha_scale : float
        n_scale     : float
        """
        b = self.base_dims
        p = pred_dims


        def _get(dims, *keys):
            for k in keys:
                v = dims.get(k)
                if v is not None:
                    return max(1,v)
            return 1

        dp_b = _get(b, "dp", "DP")
        mp_b = _get(b, "mp", "MP")
        pp_b = _get(b, "pp", "PP")
        ep_b = _get(b, "ep", "EP")
        mb_b = _get(b, "mb", "MB")
        vpp_b = _get(b, "vpp", "VPP")

        # For target: fall back to base value if dimension not specified
        dp_t = _get(p, "dp", "DP") if any(k in p for k in ("dp", "DP")) else dp_b
        mp_t = _get(p, "mp", "MP") if any(k in p for k in ("mp", "MP")) else mp_b
        pp_t = _get(p, "pp", "PP") if any(k in p for k in ("pp", "PP")) else pp_b
        ep_t = _get(p, "ep", "EP") if any(k in p for k in ("ep", "EP")) else ep_b
        mb_t = _get(p, "mb", "MB") if any(k in p for k in ("mb", "MB")) else mb_b
        vpp_t = _get(p, "vpp", "VPP") if any(k in p for k in ("vpp", "VPP")) else vpp_b

        # Actual measured base group size from IR (> 0 means reliable)
        measured_p_base = f.get("group_size", -1) if f else -1

        if lane == "DP_COMM":
            p_base      = measured_p_base if measured_p_base > 0 else dp_b
            alpha_scale = _ring_hop_scale(p_base, dp_t)
            n_scale     = mp_b / mp_t

        elif lane == "MP_COMM":
            p_base      = measured_p_base if measured_p_base > 0 else mp_b
            alpha_scale = _ring_hop_scale(p_base, mp_t)
            n_scale     = mp_b / mp_t

        elif lane == "EP_COMM":
            # AllToAll: single-round, latency independent of degree
            alpha_scale = 1.0
            n_scale     = (ep_b * dp_b) / (ep_t * dp_t)

        elif lane == "PP_COMM":
            alpha_scale = 1.0
            n_scale     = 1.0

        elif lane == "COMPUTE":
            alpha_scale = 1.0
            if pass_type == "recompute":
                # Recompute events scale with both mb and pp stage count
                n_scale = (mp_t / mp_b) * (mb_b / mb_t)
            else:
                # forward / backward: work scales with microbatch size
                n_scale = (mb_t / mb_b) * (mp_b / mp_t)

        elif lane == "BUBBLE":
            alpha_scale = (
                (pp_t - 1) / max(1, pp_b - 1)
                if pp_b > 1
                else float(pp_t > 1)
            )
            n_scale = 1.0   # β = ∞ for BUBBLE; n_scale is unused

        else:
            # UNKNOWN_COMM and any future lanes: pass through unchanged
            alpha_scale = 1.0
            n_scale     = 1.0

        return float(alpha_scale), float(n_scale)

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #

    def _print_breakdown(self, pred_dims, total_time, breakdown):
        """Print a human-readable prediction breakdown."""
        bt  = breakdown["bucket_times"]
        ot  = breakdown["overlap_terms"]
        lc  = breakdown["low_confidence_lanes"]
        unk = breakdown["unknown_comm_µs"]

        print("\n[predictor] ── Prediction breakdown ──────────────────────")
        print(f"[predictor] Target dims: {pred_dims}")
        print()

        lane_order = ["COMPUTE", "BUBBLE", "MP_COMM", "DP_COMM", "EP_COMM", "PP_COMM"]
        for lane in lane_order:
            print(f"[predictor] {lane} : {bt[lane]}")
        '''
        for lane in lane_order:
            t      = bt.get(lane, 0.0)
            conf   = " ⚠ low-confidence" if lane in lc else ""
            print(f"[predictor]   {lane:<14} {t:>12.1f} µs{conf}")
            if lane in lc:
                for reason in lc[lane][:3]:   # cap at 3 to avoid log spam
                    print(f"[predictor]     ↳ {reason}")
                if len(lc[lane]) > 3:
                    print(f"[predictor]     ↳ … and {len(lc[lane]) - 3} more")
        '''
        if unk > 0:
            print(
                    f"[predictor] UNKNOWN_COMM : {unk}"
            )
    
        '''
        print()
        print(f"[predictor] Overlap terms:")
        print(f"[predictor]   execution  (COMPUTE+BUBBLE)/vpp  {ot['execution']:>12.1f} µs")
        print(f"[predictor]   blocking   MP_COMM×{self.coeffs['mp']:.2f}          {ot['blocking']:>12.1f} µs")
        print(f"[predictor]   async      max(DP,EP)×coeff      {ot['async']:>12.1f} µs")
        print(f"[predictor]   ─────────────────────────────────────────────")
        print(f"[predictor]   total predicted step time         {total_time:>12.1f} µs")
        print(f"[predictor]                                     {total_time / 1e6:>12.4f} s")
        '''
        print()


# ─────────────────────────────────────────────────────────────────────── #
#  Module-level helpers                                                    #
# ─────────────────────────────────────────────────────────────────────── #

def _ring_hop_scale(p_base, p_target):
    """
    Latency scale factor for a ring-based collective when group size changes.

    In a ring of p ranks the startup cost is proportional to (p − 1)
    synchronisation steps:

        scale = (p_target − 1) / max(1, p_base − 1)

    Edge cases
    ──────────
    p_target == 1 : no collective required at target  → scale = 0.0
    p_base   == 1 : collective absent in base, present in target
                    → cannot extrapolate; treat base latency as representative
                    → scale = 1.0
    """
    if p_target <= 1:
        return 0.0
    if p_base <= 1:
        return 1.0
    return (p_target - 1) / (p_base - 1)
