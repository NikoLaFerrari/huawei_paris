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
    n_scale = (pp_target / pp_base) × (mb_target / mb_base)
mp is NOT a factor: each TP rank replays its own shard, and the shard
size is already baked into n_mean.

Overlap model (step-level combination)
───────────────────────────────────────
    T_total = T_execution + T_blocking + T_async

    T_execution = (T_COMPUTE + T_BUBBLE) / vpp

    T_blocking  = T_MP_COMM × coeff_mp

    T_async     = max(
                    max(dp_comm×coeff_dp, ep_comm×coeff_ep),
                    NIC_saturation_floor          # if nic_bw_gbps is set
                  )

    T_PP_COMM is NOT added (embedded in T_BUBBLE; adding it would double-count).
    UNKNOWN_COMM is tracked but not added (overlap properties unknown).

coeff_* ∈ [0, 1] are calibrated by Handler.calibrate_coefficients().

NIC bandwidth ceiling
─────────────────────
When nic_bw_gbps is provided, the total bytes transferred by all async
collective lanes (DP + EP) is divided by the physical NIC bandwidth to
obtain a saturation floor in µs.  The async term is then:
    T_async = max(overlap_model_result, saturation_floor)

This prevents unrealistically optimistic predictions for configs that
saturate the NIC when DP and EP are both active simultaneously.
"""

from collections import defaultdict

from paradise.dimensions import ALL_DIMS as AD


class Predictor:
    """Predicts step execution time for a target parallel configuration."""

    _DEFAULT_COEFFS = {"mp": 0.95, "dp": 0.10, "ep": 0.40}

    def __init__(self, model, base_dims, coeffs=None, nic_bw_gbps=None):
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

        nic_bw_gbps : float | None
            Physical NIC bandwidth ceiling in GB/s.
            When set, the combined async collective traffic (DP + EP) is
            floored at total_bytes / nic_bw to prevent predictions that
            are optimistically fast when both lanes saturate the NIC.
            Typical values: ~400 GB/s (intra-node HCCS), ~25 GB/s (inter-node RoCE).
            None = no ceiling applied.
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
                "bucket_times":         {lane: float},
                "primitive_times":      {part_key: float},
                "low_confidence_lanes": {lane: [reason_str, ...]},
                "unknown_comm_µs":      float,
                "overlap_terms": {
                    "execution":        float,
                    "blocking":         float,
                    "async":            float,
                    "nic_cap_applied":  bool,
                },
            }
        """
        vpp = max(1, pred_dims.get("vpp", pred_dims.get("VPP", 1)))

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

        # ── Overlap model ──────────────────────────────────────────────
        compute = bucket_times.get("COMPUTE", 0.0)
        bubble  = bucket_times.get("BUBBLE",  0.0)
        mp_comm = bucket_times.get("MP_COMM", 0.0)
        dp_comm = bucket_times.get("DP_COMM", 0.0)
        ep_comm = bucket_times.get("EP_COMM", 0.0)
        # PP_COMM tracked but not added — embedded in BUBBLE.
        # UNKNOWN_COMM tracked but not added — overlap properties unknown.

        execution = (compute + bubble) / vpp
        blocking  = mp_comm * self.coeffs["mp"]

        dp_eff    = dp_comm * self.coeffs["dp"]
        ep_eff    = ep_comm * self.coeffs["ep"]
        async_raw = max(dp_eff, ep_eff)

        # ── NIC bandwidth ceiling ──────────────────────────────────────
        # DP and EP both consume NIC bandwidth. When both are active they
        # compete for the same physical link. Without a floor the model
        # treats them as fully independent and may predict unrealistically
        # short async tails for configs that saturate the NIC.
        nic_cap_applied = False
        if self.nic_bw_gbps is not None and self.nic_bw_gbps > 0:
            nic_bw_bytes_per_µs = self.nic_bw_gbps * 1e3  # GB/s → bytes/µs

            def _lane_bytes(lane_prefix):
                """Recover total bytes transferred by a lane from primitive_times."""
                total_bytes = 0.0
                for pk, pf in self.model.items():
                    if pk.split("::")[0] != lane_prefix:
                        continue
                    beta  = pf.get("beta", float("inf"))
                    t     = primitive_times.get(pk, 0.0)
                    if beta not in (float("inf"),) and beta > 0:
                        alpha = pf.get("alpha", 0.0)
                        count = pf.get("count_per_step", 1.0)
                        t_bw  = max(0.0, t - alpha * count)
                        total_bytes += t_bw * beta
                return total_bytes

            total_async_bytes = _lane_bytes("DP_COMM") + _lane_bytes("EP_COMM")
            if total_async_bytes > 0:
                saturation_floor = total_async_bytes / nic_bw_bytes_per_µs
                if saturation_floor > async_raw:
                    async_raw       = saturation_floor
                    nic_cap_applied = True
                    print(
                        f"[predictor] NIC ceiling applied: "
                        f"saturation floor = {saturation_floor:.1f} µs "
                        f"({total_async_bytes/1e6:.1f} MB @ {self.nic_bw_gbps} GB/s)"
                    )

        async_net  = async_raw
        total_time = execution + blocking + async_net

        breakdown = {
            "bucket_times":         bucket_times,
            "primitive_times":      primitive_times,
            "low_confidence_lanes": dict(low_conf_lanes),
            "unknown_comm_µs":      bucket_times.get("UNKNOWN_COMM", 0.0),
            "overlap_terms": {
                "execution":       execution,
                "blocking":        blocking,
                "async":           async_net,
                "nic_cap_applied": nic_cap_applied,
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

            T = count_per_step × (α_eff + n_eff × beta_scale / β)

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
        alpha_scale, n_scale, beta_scale = self._compute_scales(
            lane, pred_dims, f, pass_type, variable
        )

        # Zero-out comm lanes where target config has no parallelism.
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
            # Latency-only primitive (BUBBLE or degenerate fit).
            time_per_call = alpha_eff
        else:
            # Full Hockney: startup + bandwidth-limited transfer.
            # beta_scale corrects for the ring efficiency (p−1)/p factor
            # changing when the collective group size changes.
            time_per_call = alpha_eff + (n_eff * beta_scale) / beta

        return max(0.0, time_per_call * count_per_step)

    # ------------------------------------------------------------------ #
    #  Scaling rules                                                       #
    # ------------------------------------------------------------------ #

    def _compute_scales(self, lane, pred_dims, f, pass_type, variable):
        """
        Return (alpha_scale, n_scale, beta_scale) for a primitive.

        alpha_scale : multiplier on the fitted α (startup latency).
        n_scale     : multiplier on n_mean (data volume).
        beta_scale  : multiplier on the n/β bandwidth term correcting for
                      the ring efficiency factor (p−1)/p changing with p.
                      Always 1.0 for non-ring lanes.

        Scaling physics per lane
        ─────────────────────────

        DP_COMM  — data-parallel gradient AllReduce (ring)
            α_scale    : _ring_hop_scale(p_base, dp_target)
            n_scale    : mp_base / mp_target  (gradient shard ∝ 1/mp)
            beta_scale : _ring_bw_scale(p_base, dp_target)

        MP_COMM  — model-parallel AllReduce / AllGather / ReduceScatter (ring)
            α_scale    : _ring_hop_scale(p_base, mp_target)
            n_scale    : mp_base / mp_target  (activation shard ∝ 1/mp)
            beta_scale : _ring_bw_scale(p_base, mp_target)

        EP_COMM  — expert-parallel AllToAll (single round, not a ring)
            α_scale    : 1.0
            n_scale    : (ep_base × dp_base) / (ep_target × dp_target)
            beta_scale : 1.0

        PP_COMM  — pipeline point-to-point Send/Receive
            α_scale    : 1.0
            n_scale    : 1.0
            beta_scale : 1.0

        COMPUTE (forward / backward)
            α_scale    : 1.0
            n_scale    : (mb_target / mb_base) × (mp_base / mp_target)
                         Work ∝ mb; per-rank shard ∝ 1/mp.
            beta_scale : 1.0

        COMPUTE (recompute)
            α_scale    : 1.0
            n_scale    : (pp_target / pp_base) × (mb_target / mb_base)
                         Event count ∝ pp (one replay per stage per backward).
                         Work per event ∝ mb.
                         mp is NOT a factor — shard size already in n_mean.
            beta_scale : 1.0

        BUBBLE
            α_scale    : (pp_target − 1) / max(1, pp_base − 1)
            n_scale    : 1.0  (β = ∞, unused)
            beta_scale : 1.0

        UNKNOWN_COMM
            1.0, 1.0, 1.0
        """
        b = self.base_dims
        p = pred_dims

        def _get(dims, *keys):
            for k in keys:
                v = dims.get(k)
                if v is not None:
                    return max(1, v)
            return 1

        dp_b  = _get(b, "dp",  "DP")
        mp_b  = _get(b, "mp",  "MP")
        pp_b  = _get(b, "pp",  "PP")
        ep_b  = _get(b, "ep",  "EP")
        mb_b  = _get(b, "mb",  "MB")

        dp_t  = _get(p, "dp",  "DP")  if any(k in p for k in ("dp",  "DP"))  else dp_b
        mp_t  = _get(p, "mp",  "MP")  if any(k in p for k in ("mp",  "MP"))  else mp_b
        pp_t  = _get(p, "pp",  "PP")  if any(k in p for k in ("pp",  "PP"))  else pp_b
        ep_t  = _get(p, "ep",  "EP")  if any(k in p for k in ("ep",  "EP"))  else ep_b
        mb_t  = _get(p, "mb",  "MB")  if any(k in p for k in ("mb",  "MB"))  else mb_b

        # Actual measured base group size from IR (> 0 means reliable)
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
            n_scale     = 1.0

        elif lane == "COMPUTE":
            alpha_scale = 1.0
            beta_scale  = 1.0
            if pass_type == "recompute":
                # Event count ∝ pp (more stages → more checkpoint replays).
                # Work per event ∝ mb.
                # mp is NOT a factor: each rank replays its own shard,
                # and that shard size is already baked into n_mean.
                n_scale = (pp_t / pp_b) * (mb_t / mb_b)
            else:
                # forward / backward: work ∝ mb, inversely ∝ mp (per-rank shard).
                n_scale = (mb_t / mb_b) * (mp_b / mp_t)

        elif lane == "BUBBLE":
            alpha_scale = (
                (pp_t - 1) / max(1, pp_b - 1)
                if pp_b > 1
                else float(pp_t > 1)
            )
            beta_scale = 1.0
            n_scale    = 1.0   # β = ∞ for BUBBLE; n_scale is unused

        else:
            # UNKNOWN_COMM and any future lanes: pass through unchanged.
            alpha_scale = 1.0
            beta_scale  = 1.0
            n_scale     = 1.0

        return float(alpha_scale), float(n_scale), float(beta_scale)

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
            t    = bt.get(lane, 0.0)
            conf = " ⚠ low-confidence" if lane in lc else ""
            print(f"[predictor]   {lane:<14} {t:>12.1f} µs{conf}")
            if lane in lc:
                for reason in lc[lane][:3]:
                    print(f"[predictor]     ↳ {reason}")
                if len(lc[lane]) > 3:
                    print(f"[predictor]     ↳ … and {len(lc[lane]) - 3} more")

        if unk > 0:
            print(
                f"[predictor]   {'UNKNOWN_COMM':<14} {unk:>12.1f} µs  (excluded from total)"
            )

        nic_note = "  ← NIC ceiling active" if ot.get("nic_cap_applied") else ""
        print()
        print(f"[predictor]   ──────────────────────────────────────────────────")
        print(f"[predictor]   total predicted step time         {total_time:>12.1f} µs")
        print(f"[predictor]                                     {total_time / 1e6:>12.4f} s")
        print()


# ─────────────────────────────────────────────────────────────────────── #
#  Module-level helpers                                                    #
# ─────────────────────────────────────────────────────────────────────── #

def _ring_hop_scale(p_base, p_target):
    """
    Alpha (latency) scale factor for a ring collective when group size changes.

    In a ring of p ranks the startup cost is proportional to (p − 1)
    synchronisation steps:

        scale = (p_target − 1) / (p_base − 1)

    Edge cases
    ──────────
    p_target == 1  →  no collective at target              →  0.0
    p_base   == 1  →  collective absent in base trace      →  1.0
    """
    if p_target <= 1:
        return 0.0
    if p_base <= 1:
        return 1.0
    return (p_target - 1) / (p_base - 1)


def _ring_bw_scale(p_base, p_target):
    """
    Bandwidth (throughput) scale factor for a ring collective.

    For a ring AllReduce / AllGather / ReduceScatter the per-byte cost is:

        T_bw(n, p) = n × (p − 1) / (p × β_wire)

    The Interpreter fits T = α + n/β at p_base, which encodes:

        1/β_fitted = (p_base − 1) / (p_base × β_wire)

    At p_target the same wire gives:

        1/β_target = (p_target − 1) / (p_target × β_wire)

    The ratio (beta_scale) corrects the fitted β for the new group size:

        beta_scale = (1/β_target) / (1/β_fitted)
                   = [(p_target − 1) × p_base] / [(p_base − 1) × p_target]

    Applied in _predict_primitive as:
        time_per_call = α_eff + (n_eff × beta_scale) / β_fitted

    Edge cases — same as _ring_hop_scale:
    p_target == 1  →  0.0
    p_base   == 1  →  1.0
    """
    if p_target <= 1:
        return 0.0
    if p_base <= 1:
        return 1.0
    return ((p_target - 1) * p_base) / ((p_base - 1) * p_target)
