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
Predictor: extrapolates per-lane or per-primitive time estimates to a target
parallel configuration.

Two prediction paths
────────────────────
1. EWA path  (preferred when lane_totals are available)
   ──────────────────────────────────────────────────────────────────────
   Input: per-lane causal event duration totals from _get_classification().
   predict_from_lane_totals() scales each lane total to target dims using
   _compute_scales physics, then applies the overlap model.

2. Hockney path  (default when formula/arch are available)
   ──────────────────────────────────────────────────────────────────────
   Input: per-primitive Hockney formula (α, β) from Interpreter.
   predict_with_breakdown() evaluates T(n) per primitive and sums per lane.

   Analytical n prediction (preferred when arch is available)
   ──────────────────────────────────────────────────────────
   When the Predictor is initialised with arch (model architecture dict)
   AND the formula carries an op_type per primitive, n for the target config
   is computed analytically from architecture parameters rather than scaling
   the base n by heuristic ratios.

   This is the key improvement over the previous _compute_scales approach:
   instead of T_target ≈ T_base × scaling_ratios (errors multiply), we
   compute n_target from first principles and evaluate T(n_target) directly.

   When arch is not available, the legacy _compute_scales path is used.

Hockney model
─────────────
    T(n) = α + n / β

    α  [µs]               startup latency
    β  [unit / µs]        effective bandwidth / throughput
    n  [bytes | FLOPs]    work metric per call

    Total per step = count_per_step × T(n_per_call)

GBS-aware scaling in _compute_scales (EWA path)
────────────────────────────────────────────────
GBS does not scale all lanes equally:

  Lanes that scale with GBS (fire per microbatch — more GBS = more work):
    COMPUTE, MP_COMM, PP_COMM, EP_COMM, IDLE
    scale factor: gbs_t / gbs_b

  Lanes that do NOT scale with GBS (fire once per step):
    DP_COMM : gradient AllReduce size = param_count / mp  (GBS-independent)
    BUBBLE  : pipeline bubble = (pp-1) microbatch slots at startup/teardown
              — fixed in absolute µs regardless of total microbatch count
    OPTIMIZER_SWAP : weight offload = param_count  (GBS-independent)

Group-size-aware alpha AND bandwidth scaling (ring collectives)
────────────────────────────────────────────────────────────────
    α_eff = α × (p_target − 1) / (p_base − 1)       [_ring_hop_scale]
    beta_scale = [(p_t−1)×p_b] / [(p_b−1)×p_t]       [_ring_bw_scale]

Overlap model (step-level combination)
───────────────────────────────────────
    T = (COMPUTE + BUBBLE)
      + PP_COMM + MP_COMM × coeff_mp
      + max(DP_COMM × coeff_dp, EP_COMM × coeff_ep)
      + OPTIMIZER_SWAP

coeff_* ∈ [0, 1] calibrated by Handler.calibrate_coefficients().
"""

from collections import defaultdict


class Predictor:
    """Predicts step execution time for a target parallel configuration."""

    _DEFAULT_COEFFS = {"mp": 0.95, "dp": 0.10, "ep": 0.40}

    def __init__(self, model, base_dims, coeffs=None, nic_bw_gbps=None, arch=None):
        """
        Parameters
        ----------
        model : dict[str, dict]
            Per-partition Hockney parameters from Interpreter.run_interpreter().
            May be empty ({}) when using the EWA prediction path exclusively.

        base_dims : dict
            Parallel dimensions of the base training config.
            Expected keys: dp, mp, pp, ep, mb, gbs, vpp.

        coeffs : dict, optional
            Non-overlapped fractions {"mp": float, "dp": float, "ep": float}.

        nic_bw_gbps : float | None
            Physical NIC bandwidth ceiling in GB/s. None = no ceiling.

        arch : dict | None
            Model architecture from Extractor._extract_model_arch().
            When provided, enables analytical n prediction for comm primitives —
            n_target is computed from architecture + target_dims rather than
            scaled from n_base by heuristic ratios.
        """
        self.model       = model
        self.base_dims   = base_dims
        self.coeffs      = coeffs if coeffs is not None else dict(self._DEFAULT_COEFFS)
        self.nic_bw_gbps = nic_bw_gbps
        self.arch        = arch          # model architecture dict (may be None)

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
        Hockney path.

        Returns
        -------
        total_time : float
        breakdown  : dict with keys bucket_times, primitive_times,
                     low_confidence_lanes, unknown_comm_µs, overlap_terms.
        """
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

            if f.get("low_confidence", False) and primitive_time > 0:
                reason = f.get("confidence_reason", "unknown reason")
                low_conf_lanes[lane].append(
                    f"{part_key.split('::', 1)[1]}: {reason}"
                )

        total_time, overlap_terms = self._apply_overlap_model(
            bucket_times, pred_dims, primitive_times
        )

        breakdown = {
            "bucket_times":         bucket_times,
            "primitive_times":      primitive_times,
            "low_confidence_lanes": dict(low_conf_lanes),
            "unknown_comm_µs":      bucket_times.get("UNKNOWN_COMM", 0.0),
            "overlap_terms":        overlap_terms,
        }

        self._print_breakdown(pred_dims, total_time, breakdown)
        return float(total_time), breakdown

    def predict_from_lane_totals(self, base_cd, pred_dims):
        """
        Scale per-lane absolute event-duration totals from a base config to
        pred_dims, then apply the overlap model.

        EWA path.  Each lane total is multiplied by a single physics-derived
        scalar from _ewa_lane_scale().  This is correct for EWA because the
        measured total is one number — decomposing it into Hockney α and n/β
        and rescaling them independently would be wrong.

        Parameters
        ----------
        base_cd : dict {lane: µs}
            Per-lane sums from _get_classification(), normalised per step.
        pred_dims : dict
            Target parallel dimensions.  Must contain at least dp/mp/pp; gbs
            and mb are used for GBS/microbatch-aware scaling when present.

        Returns
        -------
        total_time : float   overlap-adjusted predicted step time (µs)
        lane_times : dict    {lane: scaled µs} before overlap adjustment
        """
        lane_times = {}
        for lane, base_us in base_cd.items():
            if lane in ("UNKNOWN_COMM", "OPTIMIZER_SWAP", "IDLE"):
                lane_times[lane] = base_us
                continue
            scale = self._ewa_lane_scale(lane, pred_dims)
            lane_times[lane] = base_us * scale

        total_time, _ = self._apply_overlap_model(lane_times, pred_dims)
        return total_time, lane_times

    def _ewa_lane_scale(self, lane, pred_dims):
        """
        Single scalar multiplier for the EWA prediction path.

        Derived from first principles for each lane.  All quantities refer to
        the total accumulated time for that lane *per step*.

        Key identity used throughout:
            num_microbatches  =  GBS / (DP × MB)

        Lane physics
        ────────────
        COMPUTE
            Total work per rank per step = GBS × FLOPs_per_token / (DP × mp)
            MB cancels (num_mb × MB = GBS/DP).
            scale = (dp_b × mp_b) / (dp_t × mp_t) × (gbs_t / gbs_b)

        BUBBLE
            T_bubble = (pp − 1) × T_single_microbatch_forward
            T_single_mb ∝ MB × seq × H² / mp   (one microbatch through one stage)
            scale = (pp_t−1)/(pp_b−1) × (mb_t / mb_b) × (mp_b / mp_t)
            Note: DP does NOT appear — DP parallelism doesn't affect per-microbatch
            compute time.

        MP_COMM  (AllGather / ReduceScatter, per layer per microbatch)
            n_per_call  ∝ H² / mp   (weight shard; fixed per call)
            count_per_step = num_mb × layers × calls_per_layer
                           ∝ GBS / (DP × MB)
            Bandwidth dominates for large models:
            scale ≈ num_mb_ratio × (mp_b / mp_t) × ring_bw(mp_b, mp_t)
            where num_mb_ratio = (gbs_t × dp_b × mb_b) / (gbs_b × dp_t × mb_t)

        PP_COMM  (Send / Receive, once per microbatch per stage boundary)
            n_per_call  ∝ MB × seq × H     (activation tensor, one microbatch)
            count_per_step = num_mb × (pp − 1) × 2  (fwd + bwd)
            Total bandwidth ∝ (GBS/(DP×MB)) × MB × (pp−1) × H
                            = GBS × (pp−1) × H / DP
            MB cancels in the bandwidth-dominated regime.
            scale = (pp_t−1)/(pp_b−1) × (dp_b / dp_t) × (gbs_t / gbs_b)

        EP_COMM  (AlltoAllV, once per microbatch per MoE layer)
            n_per_call  ∝ MB × seq × top_k × H / ep
            count_per_step ∝ num_mb
            Total bandwidth ∝ GBS × top_k × H / (DP × ep)
            MB cancels.
            scale = (ep_b / ep_t) × (dp_b / dp_t) × (gbs_t / gbs_b)

        DP_COMM  (AllReduce gradients, once per step)
            n = param_count / mp   (gradient sharded by mp; GBS/MB-invariant)
            Bandwidth dominates:
            scale = (mp_b / mp_t) × ring_bw(dp_b, dp_t)
        """
        b = self.base_dims
        p = pred_dims

        dp_b  = _get(b, "dp");   mp_b  = _get(b, "mp")
        pp_b  = _get(b, "pp");   ep_b  = _get(b, "ep")
        gbs_b = _get(b, "gbs") or 1
        mb_b  = _get(b, "mb", "mbs") or 1

        dp_t  = _get(p, "dp")  if _has(p, "dp")  else dp_b
        mp_t  = _get(p, "mp")  if _has(p, "mp")  else mp_b
        pp_t  = _get(p, "pp")  if _has(p, "pp")  else pp_b
        ep_t  = _get(p, "ep")  if _has(p, "ep")  else ep_b
        gbs_t = _get(p, "gbs") if _has(p, "gbs") else gbs_b
        mb_t  = _get(p, "mb", "mbs") if _has(p, "mb", "mbs") else mb_b

        # num_microbatches ratio: how many more/fewer microbatches in target
        num_mb_ratio = (gbs_t * dp_b * mb_b) / (gbs_b * dp_t * mb_t)

        if lane == "COMPUTE":
            # Total ∝ GBS / (DP × mp); MB cancels
            return (dp_b * mp_b) / (dp_t * mp_t) * (gbs_t / gbs_b)

        elif lane == "BUBBLE":
            # Total ∝ (pp−1) × MB / mp; DP does not appear
            pp_scale = (pp_t - 1) / (pp_b - 1) if pp_b > 1 else (0.0 if pp_t <= 1 else 1.0)
            return pp_scale * (mb_t / mb_b) * (mp_b / mp_t)

        elif lane == "MP_COMM":
            # Total ∝ num_mb × (H²/mp) × ring_bw_correction
            # bandwidth-dominated approximation
            return num_mb_ratio * (mp_b / mp_t) * _ring_bw_scale(mp_b, mp_t)

        elif lane == "PP_COMM":
            # Total bandwidth ∝ GBS × (pp−1) / DP; MB cancels
            pp_scale = (pp_t - 1) / (pp_b - 1) if pp_b > 1 else (0.0 if pp_t <= 1 else 1.0)
            return pp_scale * (dp_b / dp_t) * (gbs_t / gbs_b)

        elif lane == "EP_COMM":
            # Total bandwidth ∝ GBS / (DP × ep); MB cancels
            return (ep_b / ep_t) * (dp_b / dp_t) * (gbs_t / gbs_b)

        elif lane == "DP_COMM":
            # Fires once per step; size ∝ param_count/mp; bandwidth dominated
            return (mp_b / mp_t) * _ring_bw_scale(dp_b, dp_t)

        else:
            return 1.0

    # ------------------------------------------------------------------ #
    #  Overlap model (shared by both paths)                               #
    # ------------------------------------------------------------------ #

    def _apply_overlap_model(self, lane_times, pred_dims, primitive_times=None):
        """
        Combine per-lane times into a step-time prediction.

            T = (COMPUTE + BUBBLE)
              + PP_COMM + MP_COMM × coeff_mp
              + max(DP_COMM × coeff_dp, EP_COMM × coeff_ep)
              + OPTIMIZER_SWAP

        Returns (total_time, overlap_terms_dict).
        """
        compute        = lane_times.get("COMPUTE",        0.0)
        bubble         = lane_times.get("BUBBLE",         0.0)
        mp_comm        = lane_times.get("MP_COMM",        0.0)
        dp_comm        = lane_times.get("DP_COMM",        0.0)
        ep_comm        = lane_times.get("EP_COMM",        0.0)
        pp_comm        = lane_times.get("PP_COMM",        0.0)
        optimizer_swap = lane_times.get("OPTIMIZER_SWAP", 0.0)

        execution = compute + bubble
        blocking  = pp_comm + mp_comm * self.coeffs["mp"]
        dp_eff    = dp_comm * self.coeffs["dp"]
        ep_eff    = ep_comm * self.coeffs["ep"]
        async_raw = max(dp_eff, ep_eff)

        nic_cap_applied = False
        if self.nic_bw_gbps is not None and self.nic_bw_gbps > 0 and primitive_times:
            nic_bw_bytes_per_us = self.nic_bw_gbps * 1e3

            def _lane_bytes(lane_prefix):
                total_bytes = 0.0
                for pk, pf in self.model.items():
                    if pk.split("::")[0] != lane_prefix:
                        continue
                    beta = pf.get("beta", float("inf"))
                    t    = primitive_times.get(pk, 0.0)
                    if beta not in (float("inf"),) and beta > 0:
                        alpha = pf.get("alpha", 0.0)
                        count = pf.get("count_per_step", 1.0)
                        t_bw  = max(0.0, t - alpha * count)
                        total_bytes += t_bw * beta
                return total_bytes

            total_async_bytes = _lane_bytes("DP_COMM") + _lane_bytes("EP_COMM")
            if total_async_bytes > 0:
                saturation_floor = total_async_bytes / nic_bw_bytes_per_us
                if saturation_floor > async_raw:
                    async_raw       = saturation_floor
                    nic_cap_applied = True

        idle = lane_times.get("IDLE", 0.0)
        total_time = execution + blocking + async_raw + optimizer_swap + idle

        return float(total_time), {
            "execution":       execution,
            "blocking":        blocking,
            "async":           async_raw,
            "optimizer_swap":  optimizer_swap,
            "nic_cap_applied": nic_cap_applied,
        }

    # ------------------------------------------------------------------ #
    #  Per-primitive prediction (Hockney path)                            #
    # ------------------------------------------------------------------ #

    def _predict_primitive(self, part_key, f, pred_dims):
        """
        Predict time for one Hockney partition at pred_dims.

        Analytical n path (preferred when arch available):
            1. Retrieve op_type and n_analytical_base from formula
            2. Compute n_analytical_target from arch + pred_dims
            3. n_scale = n_analytical_target / n_analytical_base
            4. T = count_per_step × alpha_scale × α + (n_base × n_scale / β) × count_per_step

        Legacy _compute_scales path (fallback when arch absent):
            Uses heuristic scaling ratios derived from config dimensions.
        """
        alpha          = f["alpha"]
        beta           = f["beta"]
        n_mean         = f["n_mean"]
        count_per_step = f["count_per_step"]
        variable       = f.get("variable", "bytes")
        pass_type      = f.get("pass_type", "unknown")
        op_type        = f.get("op_type",   "unknown")

        lane = part_key.split("::")[0]

        # ── Zero-out disabled lanes ───────────────────────────────────
        if lane == "MP_COMM" and _get_dim(pred_dims, "mp", self.base_dims) <= 1:
            return 0.0
        if lane == "DP_COMM" and _get_dim(pred_dims, "dp", self.base_dims) <= 1:
            return 0.0
        if lane == "EP_COMM" and _get_dim(pred_dims, "ep", self.base_dims) <= 1:
            return 0.0

        # ── Determine n_scale ─────────────────────────────────────────
        # Analytical path: use arch + op_type to compute n for target config
        n_analytical_base = f.get("n_analytical_base")
        if (self.arch is not None
                and op_type not in ("unknown", None)
                and n_analytical_base is not None
                and n_analytical_base > 0
                and lane not in ("COMPUTE", "BUBBLE")):
            n_analytical_target = _analytical_comm_n(op_type, self.arch, pred_dims)
            if n_analytical_target is not None and n_analytical_target > 0:
                n_scale_comm = n_analytical_target / n_analytical_base
                alpha_scale, _, beta_scale = self._compute_scales(
                    lane, pred_dims, f, pass_type, variable
                )
                alpha_eff     = alpha * alpha_scale
                n_eff         = n_mean * n_scale_comm
                if beta == float("inf") or beta <= 0.0:
                    time_per_call = alpha_eff
                else:
                    time_per_call = alpha_eff + (n_eff * beta_scale) / beta
                return max(0.0, time_per_call * count_per_step)

        # ── Legacy _compute_scales path ───────────────────────────────
        alpha_scale, n_scale, beta_scale = self._compute_scales(
            lane, pred_dims, f, pass_type, variable
        )

        alpha_eff = alpha * alpha_scale
        n_eff     = n_mean * n_scale

        if beta == float("inf") or beta <= 0.0:
            time_per_call = alpha_eff
        else:
            time_per_call = alpha_eff + (n_eff * beta_scale) / beta

        return max(0.0, time_per_call * count_per_step)

    # ------------------------------------------------------------------ #
    #  Scaling rules                                                       #
    # ------------------------------------------------------------------ #

    def _compute_scales(self, lane, pred_dims, f, pass_type, variable):
        """
        Return (alpha_scale, n_scale, beta_scale) for a lane.

        Used by the EWA path (predict_from_lane_totals) and as the fallback
        for the Hockney path when arch is unavailable.

        Scaling physics
        ───────────────
        GBS scaling applies only to lanes that fire per microbatch:
            COMPUTE, MP_COMM, PP_COMM, EP_COMM, IDLE → ×(gbs_t/gbs_b)
        GBS-invariant lanes (fire once per step):
            DP_COMM  → no GBS factor (gradient size = param_count/mp)
            BUBBLE   → no GBS factor (bubble = fixed (pp-1) microbatch slots)
            OPTIMIZER_SWAP → no GBS factor (weight size = param_count)

        Per lane:
          DP_COMM  :  α×ring_hop(dp), n×(mp_b/mp_t),                        β×ring_bw(dp)
          MP_COMM  :  α×ring_hop(mp), n×(mp_b/mp_t)×(gbs_t/gbs_b),         β×ring_bw(mp)
          EP_COMM  :  α×1,            n×(ep_b/ep_t)×(dp_b/dp_t)×(gbs_t/gbs_b), β×1
          PP_COMM  :  α×1,            n×(mp_b/mp_t)×(gbs_t/gbs_b),              β×1
          COMPUTE (fwd/bwd):   n×(dp_b×mp_b)/(dp_t×mp_t)×(gbs_t/gbs_b)
          COMPUTE (recompute): n×(pp_t/pp_b)×(dp_b×mp_b)/(dp_t×mp_t)×(gbs_t/gbs_b)
          BUBBLE:  α×(pp_t-1)/(pp_b-1)×(mp_b/mp_t)×(dp_b/dp_t), n×1 (β=∞)
          others:  1, 1, 1
        """
        b = self.base_dims
        p = pred_dims

        dp_b = _get(b, "dp")
        mp_b = _get(b, "mp")
        pp_b = _get(b, "pp")
        ep_b = _get(b, "ep")
        gbs_b = _get(b, "gbs") or 1

        dp_t = _get(p, "dp") if _has(p, "dp") else dp_b
        mp_t = _get(p, "mp") if _has(p, "mp") else mp_b
        pp_t = _get(p, "pp") if _has(p, "pp") else pp_b
        ep_t = _get(p, "ep") if _has(p, "ep") else ep_b
        gbs_t = _get(p, "gbs") if _has(p, "gbs") else gbs_b

        gbs_scale = gbs_t / gbs_b if gbs_b > 0 else 1.0

        measured_p_base = (f.get("group_size", -1) if f else -1)

        if lane == "DP_COMM":
            # AllReduce fires once per step — GBS does not affect gradient size
            p_base      = measured_p_base if measured_p_base > 0 else dp_b
            alpha_scale = _ring_hop_scale(p_base, dp_t)
            beta_scale  = _ring_bw_scale(p_base, dp_t)
            n_scale     = mp_b / mp_t          # gradient sharded by mp; no GBS factor

        elif lane == "MP_COMM":
            # AllGather/ReduceScatter per layer per microbatch
            # count_per_step ∝ num_mb = GBS/(DP×MB); n_per_call ∝ H²/mp
            mb_b        = _get(b, "mb", "mbs") or 1
            mb_t        = _get(p, "mb", "mbs") if _has(p, "mb", "mbs") else mb_b
            p_base      = measured_p_base if measured_p_base > 0 else mp_b
            alpha_scale = _ring_hop_scale(p_base, mp_t)
            beta_scale  = _ring_bw_scale(p_base, mp_t)
            # n_scale captures: per-call size (mp_b/mp_t) × count (gbs,dp,mb)
            n_scale     = (mp_b / mp_t) * gbs_scale * (dp_b / dp_t) * (mb_b / mb_t)

        elif lane == "EP_COMM":
            # AlltoAllV per microbatch → scales with tokens per step
            alpha_scale = 1.0
            beta_scale  = 1.0
            # n_per_call = tokens_per_mb × top_k × H / ep  (tokens_per_mb fixed)
            # count_per_step scales with dp_b/dp_t (fewer dp → more microbatches)
            # total: n_scale = (ep_b/ep_t) × (dp_b/dp_t) × (gbs_t/gbs_b)
            n_scale     = (ep_b / ep_t) * (dp_b / dp_t) * gbs_scale

        elif lane == "PP_COMM":
            # Send/Receive per microbatch; n_per_call ∝ MB×seq×H; MB cancels
            # Total bandwidth ∝ GBS × (pp−1) × H / DP
            pp_scale    = (pp_t - 1) / (pp_b - 1) if pp_b > 1 else (0.0 if pp_t <= 1 else 1.0)
            alpha_scale = 1.0
            beta_scale  = 1.0
            n_scale     = pp_scale * (dp_b / dp_t) * gbs_scale

        elif lane == "COMPUTE":
            alpha_scale = 1.0
            beta_scale  = 1.0
            if pass_type == "recompute":
                # Recompute also scales with pp ratio (stages recompute forward)
                n_scale = (pp_t / pp_b) * (dp_b * mp_b) / (dp_t * mp_t) * gbs_scale
            else:
                n_scale = (dp_b * mp_b) / (dp_t * mp_t) * gbs_scale

        elif lane == "BUBBLE":
            # Bubble = (pp-1) × T_single_microbatch ∝ (pp-1) × MB / mp
            # DP does NOT appear — DP doesn't change per-microbatch time.
            mb_b        = _get(b, "mb", "mbs") or 1
            mb_t        = _get(p, "mb", "mbs") if _has(p, "mb", "mbs") else mb_b
            pp_scale    = (pp_t - 1) / (pp_b - 1) if pp_b > 1 else (0.0 if pp_t <= 1 else 1.0)
            alpha_scale = pp_scale * (mb_t / mb_b) * (mp_b / mp_t)
            beta_scale  = 1.0
            n_scale     = 1.0           # BUBBLE is latency-only; n_scale unused

        else:
            alpha_scale = 1.0
            beta_scale  = 1.0
            n_scale     = 1.0

        return float(alpha_scale), float(n_scale), float(beta_scale)

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #

    def _print_breakdown(self, pred_dims, total_time, breakdown):
        bt  = breakdown["bucket_times"]
        lc  = breakdown.get("low_confidence_lanes", {})
        unk = breakdown.get("unknown_comm_µs", 0.0)

        print("\n[predictor] ── Prediction breakdown ──────────────────────")
        print(f"[predictor] Target dims: {pred_dims}")
        print()

        for lane in ["COMPUTE", "BUBBLE", "PP_COMM", "MP_COMM",
                     "DP_COMM", "EP_COMM", "OPTIMIZER_SWAP"]:
            t = bt.get(lane, 0.0)
            if t == 0.0:
                continue
            conf = " ⚠ low-confidence" if lane in lc else ""
            print(f"[predictor]   {lane:<16} {t:>12.1f} µs{conf}")
        if unk > 0:
            print(f"[predictor]   {'UNKNOWN_COMM':<16} {unk:>12.1f} µs  (excluded)")

        print()
        print(f"[predictor]   {'─'*50}")
        print(f"[predictor]   total predicted              {total_time:>12.1f} µs")
        print(f"[predictor]                                {total_time / 1e6:>12.4f} s")
        print()


# ─────────────────────────────────────────────────────────────────────── #
#  Module-level helpers                                                    #
# ─────────────────────────────────────────────────────────────────────── #

def _get(dims, *keys):
    """
    Return max(1, value) for the first matching key in dims, or 1.
    Checks each key case-insensitively so that uppercase ND strategy dicts
    (e.g. {DP: 4, MP: 2}) and lowercase normalised sample dicts both work.
    """
    for k in keys:
        for variant in (k, k.upper(), k.lower()):
            v = dims.get(variant)
            if v is not None:
                return max(1, int(v))
    return 1


def _has(dims, *keys):
    """Return True if any key is present in dims (case-insensitive)."""
    for k in keys:
        for variant in (k, k.upper(), k.lower()):
            if variant in dims:
                return True
    return False


def _get_dim(pred_dims, key, base_dims):
    """Return pred_dims[key] if present, else base_dims[key], else 1. Case-insensitive."""
    for variant in (key, key.upper(), key.lower()):
        v = pred_dims.get(variant)
        if v is not None:
            return max(1, int(v))
    for variant in (key, key.upper(), key.lower()):
        v = base_dims.get(variant)
        if v is not None:
            return max(1, int(v))
    return 1


def _ring_hop_scale(p_base, p_target):
    """Alpha scale for ring collective: (p_target-1)/(p_base-1)."""
    if p_target <= 1:
        return 0.0
    if p_base <= 1:
        return 1.0
    return (p_target - 1) / (p_base - 1)


def _ring_bw_scale(p_base, p_target):
    """
    Bandwidth correction for ring efficiency factor (p-1)/p.

        beta_scale = [(p_target-1) × p_base] / [(p_base-1) × p_target]
    """
    if p_target <= 1:
        return 0.0
    if p_base <= 1:
        return 1.0
    return ((p_target - 1) * p_base) / ((p_base - 1) * p_target)


def _analytical_comm_n(op_type, arch, dims):
    """
    Compute physically correct per-call message size (bytes) for a collective
    from model architecture parameters.

    Mirrors Extractor._analytical_comm_n() — kept as a standalone function
    here to avoid importing Extractor into Predictor.

    Parameters
    ----------
    op_type : str     e.g. "AllGather", "AlltoAllV", "Send"
    arch    : dict    model architecture dict (hidden_dim, ffn_hidden_dim, ...)
    dims    : dict    normalised parallel dims (mp, dp, ep, pp, mbs/mb, ...)

    Returns
    -------
    int | None    bytes per call at the given config
    """
    if arch is None:
        return None

    H   = arch["hidden_dim"]
    F   = arch["ffn_hidden_dim"]
    L   = arch["num_layers"]
    K   = max(1, arch["top_k"])
    S   = arch["seq_len"]
    dt  = arch["dtype_bytes"]

    def _ci(d, *keys):
        for k in keys:
            for v in (d.get(k), d.get(k.upper()), d.get(k.lower())):
                if v is not None:
                    return max(1, int(v))
        return 1

    mp  = _ci(dims, "mp")
    ep  = _ci(dims, "ep")
    mbs = _ci(dims, "mbs", "mb") or max(1, arch.get("mbs", 1))

    tokens_per_mb = S * mbs

    if op_type in ("AllGather", "ReduceScatter", "AllReduce_AllGather"):
        return int(H * (H + F) // (2 * mp) * dt)

    elif op_type == "AllReduce":
        # Fires once per step; gradient size is GBS-independent
        return int(L * (4 * H * H + 3 * H * F) // mp * dt)

    elif op_type in ("AlltoAllV", "AlltoAllVC"):
        return int(tokens_per_mb * K * H // ep * dt)

    elif op_type in ("Send", "Receive"):
        return int(tokens_per_mb * H * dt)

    elif op_type == "Broadcast":
        return int(H * dt)

    return None
