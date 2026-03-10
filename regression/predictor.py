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
1. EWA path  (preferred, used when lane_totals are available)
   ──────────────────────────────────────────────────────────
   Input: per-lane actual event duration totals from _get_classification().
   These are sums of raw event durations (compute kernels + comm events +
   EVENT_WAIT + swap), normalised per step. They are NOT EWA wait-tail times
   — they are full event durations classified by lane.

   predict_from_lane_totals() applies _compute_scales physics to each lane
   total and then uses the same overlap model as the Hockney path to combine
   lanes into a step time prediction.

   Why the overlap model is still needed here
   ──────────────────────────────────────────
   DP and EP comm events execute asynchronously on the comm stream while
   compute kernels run on the compute stream. Summing all lane durations
   independently over-counts: the async lanes do not add to wall-clock time
   unless they are the critical path. The overlap model encodes exactly this.

   Why EWA wait-tail times are NOT used as lane_totals
   ─────────────────────────────────────────────────────
   EWA's summarize_wait_causes() returns the fraction of the compute stream
   that was stalled waiting for each comm lane. This is only the tail of the
   comm duration visible from the compute stream. DP and EP communications
   that complete before the compute stream reaches the sync point contribute
   zero to the EWA wait time but non-zero to actual step time. Using EWA
   wait-tail times as lane_totals would give sum(cd) ≈ 50% of step_time.

2. Hockney path  (fallback)
   ──────────────────────────────────────────────────────────────────────
   Input: per-primitive Hockney formula (α, β) from Interpreter.
   predict_with_breakdown() sums α×count + n/β×count over all primitives
   per lane and applies the same overlap model.

Hockney model
─────────────
For a single primitive invocation:

    T(n) = α + n / β

    α  [µs]               startup latency
    β  [unit / µs]        effective bandwidth (unit = bytes or FLOPs)
    n  [bytes | FLOPs]    work metric — see "variable" field in formula

The total predicted time for a primitive over one training step is:

    T_primitive = count_per_step × (α_eff + n_eff × beta_scale / β)

where α_eff, n_eff, and beta_scale are derived from config-driven scaling
(see _compute_scales).

Group-size-aware alpha AND bandwidth scaling
────────────────────────────────────────────
For ring-based collectives (DP_COMM, MP_COMM) both the startup latency α
and the per-byte bandwidth term change when group size changes.

Alpha (latency) scales with ring hops:
    α_eff = α × (p_target − 1) / (p_base − 1)      [_ring_hop_scale]

Bandwidth efficiency for a ring of p ranks:
    T_bw = n × (p − 1) / (p × β_wire)
The fitted β already encodes (p_base−1)/p_base.  When predicting at a
different group size the bandwidth term must be re-scaled:
    beta_scale = [(p_target−1) × p_base] / [(p_base−1) × p_target]   [_ring_bw_scale]

COMPUTE recompute n_scale
─────────────────────────
Recompute replays forward ops once per pipeline stage during the backward
pass.  The event count and the per-event work both change independently:
    n_scale = (pp_target / pp_base) × (dp_base × mp_base) / (dp_target × mp_target)

Overlap model (step-level combination)
───────────────────────────────────────
Used by both prediction paths:

    T_total = T_execution + T_blocking + T_async + T_optimizer_swap

    T_execution      = T_COMPUTE + T_BUBBLE
    T_blocking       = T_PP_COMM + T_MP_COMM × coeff_mp
    T_async          = max(T_DP_COMM × coeff_dp, T_EP_COMM × coeff_ep)
    T_optimizer_swap = T_OPTIMIZER_SWAP  (fully blocking weight offload)

    UNKNOWN_COMM and IDLE are tracked but excluded from the total.

coeff_* ∈ [0, 1] are calibrated by Handler.calibrate_coefficients().
"""

from collections import defaultdict


class Predictor:
    """Predicts step execution time for a target parallel configuration."""

    _DEFAULT_COEFFS = {"mp": 0.95, "dp": 0.10, "ep": 0.40}

    def __init__(self, model, base_dims, coeffs=None, nic_bw_gbps=None):
        """
        Parameters
        ----------
        model : dict[str, dict]
            Per-partition Hockney parameters from Interpreter.run_interpreter().
            May be empty ({}) when using the EWA prediction path.

        base_dims : dict
            Parallel dimensions of the training traces used for regression.
            Expected keys: dp, mp, pp, ep, mb, vpp.

        coeffs : dict, optional
            Non-overlapped fractions {"mp": float, "dp": float, "ep": float}.
            Defaults to _DEFAULT_COEFFS if None.

        nic_bw_gbps : float | None
            Physical NIC bandwidth ceiling in GB/s. None = no ceiling.
        """
        self.model       = model
        self.base_dims   = base_dims
        self.coeffs      = coeffs if coeffs is not None else dict(self._DEFAULT_COEFFS)
        self.nic_bw_gbps = nic_bw_gbps

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
        pred_dims using _compute_scales physics, then apply the overlap model.

        EWA path.

        Parameters
        ----------
        base_cd : dict {lane: µs}
            Per-lane sums of actual event durations from _get_classification(),
            normalised per step. Full comm + compute event durations — NOT
            EWA wait-tail times.

        pred_dims : dict
            Target parallel dimensions.

        Returns
        -------
        total_time : float   overlap-adjusted predicted wall-clock step time µs
        lane_times : dict    {lane: scaled µs} before overlap adjustment
        """
        lane_times = {}

        for lane, base_us in base_cd.items():
            if lane in ("UNKNOWN_COMM","OPTIMIZER"):
                lane_times[lane] = base_us
                continue

            alpha_scale, n_scale, _ = self._compute_scales(
                lane, pred_dims, None, "forward", "bytes"
            )
            lane_times[lane] = base_us * alpha_scale * n_scale

        total_time, _ = self._apply_overlap_model(lane_times, pred_dims)
        return total_time, lane_times

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

        # NIC bandwidth ceiling (Hockney path only — needs primitive_times)
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
        alpha          = f["alpha"]
        beta           = f["beta"]
        n_mean         = f["n_mean"]
        count_per_step = f["count_per_step"]
        variable       = f.get("variable", "bytes")
        pass_type      = f.get("pass_type", "unknown")

        lane = part_key.split("::")[0]
        alpha_scale, n_scale, beta_scale = self._compute_scales(
            lane, pred_dims, f, pass_type, variable
        )

        if lane == "MP_COMM":
            if pred_dims.get("mp", pred_dims.get("MP", self.base_dims.get("mp", 1))) <= 1:
                return 0.0
        elif lane == "DP_COMM":
            if pred_dims.get("dp", pred_dims.get("DP", self.base_dims.get("dp", 1))) <= 1:
                return 0.0
        elif lane == "EP_COMM":
            if pred_dims.get("ep", pred_dims.get("EP", 1)) <= 1:
                return 0.0

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

        Scaling physics per lane
        ─────────────────────────
        DP_COMM  :  α×ring_hop(dp), n×mp_b/mp_t, β×ring_bw(dp)
        MP_COMM  :  α×ring_hop(mp), n×mp_b/mp_t, β×ring_bw(mp)
        EP_COMM  :  α×1,            n×(ep_b×dp_b)/(ep_t×dp_t),   β×1
        PP_COMM  :  α×1,            n×mp_b/mp_t,                  β×1
        COMPUTE fwd/bwd  :  n×(dp_b×mp_b)/(dp_t×mp_t)
        COMPUTE recompute:  n×(pp_t/pp_b)×(dp_b×mp_b)/(dp_t×mp_t)
        BUBBLE   :  α×(pp_t-1)/(pp_b-1)×mp_b/mp_t,  n×1 (β=∞)
        others   :  1, 1, 1
        """
        b = self.base_dims
        p = pred_dims

        def _get(dims, *keys):
            for k in keys:
                v = dims.get(k)
                if v is not None:
                    return max(1, v)
            return 1

        dp_b = _get(b, "dp", "DP")
        mp_b = _get(b, "mp", "MP")
        pp_b = _get(b, "pp", "PP")
        ep_b = _get(b, "ep", "EP")

        dp_t = _get(p, "dp", "DP") if any(k in p for k in ("dp", "DP")) else dp_b
        mp_t = _get(p, "mp", "MP") if any(k in p for k in ("mp", "MP")) else mp_b
        pp_t = _get(p, "pp", "PP") if any(k in p for k in ("pp", "PP")) else pp_b
        ep_t = _get(p, "ep", "EP") if any(k in p for k in ("ep", "EP")) else ep_b

        measured_p_base = f.get("group_size", -1) if f else -1

        if lane == "DP_COMM":
            p_base      = measured_p_base if measured_p_base > 0 else dp_b
            alpha_scale = _ring_hop_scale(p_base, dp_t)
            beta_scale  = _ring_bw_scale(p_base, dp_t)
            n_scale     = mp_b / mp_t

        elif lane == "MP_COMM":
            p_base      = measured_p_base if measured_p_base > 0 else mp_b
            alpha_scale = _ring_hop_scale(p_base, mp_t)
            beta_scale  = _ring_bw_scale(p_base, mp_t)
            n_scale     = mp_b / mp_t

        elif lane == "EP_COMM":
            alpha_scale = 1.0
            beta_scale  = 1.0
            n_scale     = (ep_b * dp_b) / (ep_t * dp_t)

        elif lane == "PP_COMM":
            alpha_scale = 1.0
            beta_scale  = 1.0
            n_scale     = mp_b / mp_t

        elif lane == "COMPUTE":
            alpha_scale = 1.0
            beta_scale  = 1.0
            if pass_type == "recompute":
                n_scale = (pp_t / pp_b) * (mp_b / mp_t)#(pp_t / pp_b) * (dp_b * mp_b) / (dp_t * mp_t)
            else:
                n_scale = mp_b / mp_t #(dp_b * mp_b) / (dp_t * mp_t)

        elif lane == "BUBBLE":
            pp_scale    = (pp_t - 1) / (pp_b - 1) if pp_b > 1 else 1.0
            mp_scale    = mp_b / mp_t
            alpha_scale = pp_scale * mp_scale
            beta_scale  = 1.0
            n_scale     = 1.0

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

def _ring_hop_scale(p_base, p_target):
    """Alpha scale for ring collective: (p_target-1)/(p_base-1)."""
    if p_target <= 1:
        return 0.0
    if p_base <= 1:
        return 1.0
    return (p_target - 1) / (p_base - 1)


def _ring_bw_scale(p_base, p_target):
    """
    Bandwidth correction for ring efficiency factor (p-1)/p changing with p.

        beta_scale = [(p_target-1) × p_base] / [(p_base-1) × p_target]
    """
    if p_target <= 1:
        return 0.0
    if p_base <= 1:
        return 1.0
    return ((p_target - 1) * p_base) / ((p_base - 1) * p_target)
