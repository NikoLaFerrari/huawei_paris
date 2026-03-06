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
Interpreter: fits a Hockney model per sub-partitioned primitive.

Hockney model
─────────────
    T(n) = α + n / β

where
    α  [µs]             startup latency   (regression intercept)
    β  [unit / µs]      effective bandwidth or throughput (1 / slope)
    n  [bytes | FLOPs]  work metric — bytes for comms, FLOPs for MatMul-family

Sub-partitioning strategy
─────────────────────────
The extractor now carries dtype, alg_type, and pass_type metadata on every
data point. A single primitive key like EP_COMM::AlltoAllV may contain a mix
of BFP16 and FP32 invocations, each following a physically different Hockney
curve (different hardware bandwidth ceilings per dtype) and potentially a
different collective algorithm (MESH vs RING). Similarly, COMPUTE::MatMul
forward and backward invocations often have different durations at the same
FLOP count due to different kernel paths and memory pressure.

Mixing these sub-populations into a single OLS regression produces a
superposition curve that represents none of them accurately.

Therefore we sub-partition as follows:
  Communication primitives : lane::primitive[dtype|alg_type]
  Compute primitives        : lane::primitive[pass_type]

The predictor is unaffected — it still accumulates by lane prefix.

Outlier filtering for PP_COMM Receive
──────────────────────────────────────
PP_COMM Receive is a blocking call: it does not return until the upstream
pipeline stage delivers its activation tensor.  The measured duration
therefore includes both the actual transfer time AND any time the downstream
rank spent idle waiting for the upstream rank to finish its computation.
In practice this produces a bimodal distribution: a fast cluster around the
true transfer time (e.g. ~4–6 ms for 80 MB at BFP16) and a slow cluster
with durations up to 600 ms driven entirely by pipeline stalls.

Regressing on the mixed distribution would produce a wildly inflated α.
We use IQR-based outlier removal on y before fitting for PP_COMM primitives
to isolate the true transfer-time cluster.

Output formula schema (one entry per sub-partitioned key)
──────────────────────────────────────────────────────────
    {
        "alpha":          float,   # startup latency, µs  (≥ 0)
        "beta":           float,   # bandwidth/throughput (> 0 or ∞)
        "n_mean":         float,   # mean work metric in training data
        "variable":       str,     # "flops" | "bytes"
        "count_per_step": float,   # mean invocations per training step
        "r2":             float,   # coefficient of determination
        "group_size":     int,     # modal collective group size p (-1 if unknown)
        "pass_type":      str,     # "forward"|"backward"|"recompute"|"mixed"|"unknown"
        "dtype":          str,     # e.g. "BFP16", "FP32", "mixed"
        "alg_type":       str,     # e.g. "MESH-RING-NHR", "mixed"
        "low_confidence": bool,    # True when fit is known to be unreliable
        "confidence_reason": str,  # human-readable explanation when low_confidence
    }
"""

from collections import defaultdict
from statistics import mode as stat_mode

import numpy as np
from scipy import stats


# Lanes for which IQR outlier filtering is applied to y before fitting.
# PP_COMM Receive durations are contaminated by pipeline stall time.
_OUTLIER_FILTER_LANES = {"PP_COMM"}

# IQR multiplier for outlier removal (standard Tukey fence)
_IQR_MULTIPLIER = 1.5


class Interpreter:
    """Fits per-primitive Hockney equations to sub-partitioned trace data."""

    def __init__(self, all_samples):
        """
        Parameters
        ----------
        all_samples : list[dict]
            Output of Extractor.run_extractor().
            Each element:
            {
                "data": {
                    "lane::primitive": [
                        {
                            "x":          float,   # bytes or FLOPs
                            "y":          float,   # duration µs
                            "ts":         float,   # timestamp µs
                            "dtype":      str,
                            "alg_type":   str,
                            "pass_type":  str,
                            "group_size": int,
                            "is_fused":   bool,
                            ...
                        },
                        ...
                    ]
                },
                "dims": {...}
            }
        """
        self.all_samples = all_samples
        self.formula     = {}

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run_interpreter(self):
        """
        Main entry point.

        Returns
        -------
        self.formula : dict[str, dict]
            Per sub-partitioned-key Hockney parameters.
        """
        print("[interpreter] Fitting Hockney equations per sub-partitioned primitive …\n")

        partitions, partition_counts = self._build_partitions()
        partitions, partition_counts = self._merge_degenerate_partitions(partitions, partition_counts)
        n_samples = max(1, len(self.all_samples))

        for part_key, data in partitions.items():
            x_vals = np.array(data["x"], dtype=float)
            y_vals = np.array(data["y"], dtype=float)

            # ── Outlier filtering on y for stall-contaminated lanes ───
            lane = part_key.split("::")[0]
            if lane in _OUTLIER_FILTER_LANES and len(y_vals) >= 4:
                x_vals, y_vals, n_removed = self._iqr_filter(x_vals, y_vals)
                if n_removed > 0:
                    print(
                        f"[interpreter] {part_key}: removed {n_removed} outlier "
                        f"y-values (IQR filter, stall contamination)."
                    )

            count_per_step = partition_counts[part_key] / n_samples
            n_mean         = float(np.mean(x_vals)) if len(x_vals) > 0 else 1.0

            # ── Metadata from points ──────────────────────────────────
            variable    = data["variable"]        # "flops" | "bytes"
            group_size  = data["group_size"]
            pass_type   = data["pass_type"]
            dtype       = data["dtype"]
            alg_type    = data["alg_type"]
            any_fused   = data["any_fused"]

            # ── Hockney fit ───────────────────────────────────────────
            alpha, beta, r2, low_conf, conf_reason = self._fit_hockney(
                lane, part_key, x_vals, y_vals, any_fused
            )

            self.formula[part_key] = {
                "alpha":            alpha,
                "beta":             beta,
                "n_mean":           n_mean,
                "variable":         variable,
                "count_per_step":   count_per_step,
                "r2":               r2,
                "group_size":       group_size,
                "pass_type":        pass_type,
                "dtype":            dtype,
                "alg_type":         alg_type,
                "low_confidence":   low_conf,
                "confidence_reason":conf_reason,
            }

        self._print_summary()
        return self.formula

    # ------------------------------------------------------------------ #
    #  Partition building                                                  #
    # ------------------------------------------------------------------ #

    def _build_partitions(self):
        """
        Walk all samples and build sub-partitioned data collections.

        Sub-partition rules
        ───────────────────
        Communication (non-COMPUTE, non-BUBBLE):
            key = lane::primitive[dtype|alg_type]
            e.g. EP_COMM::AlltoAllV[BFP16|MESH-RING-NHR]

        Compute (COMPUTE lane):
            key = lane::primitive[pass_type]
            e.g. COMPUTE::MatMul[forward]

        BUBBLE:
            key = lane::primitive  (no sub-partition; duration is latency-only)

        For each partition we accumulate:
            x, y                : work metric and duration lists
            group_sizes         : to derive modal group size
            is_fused_flags      : to flag low-confidence partitions
            variable            : "flops" if x came from x_flops, else "bytes"

        Returns
        -------
        partitions       : dict[str, {x, y, variable, group_size, ...}]
        partition_counts : dict[str, int]   total raw invocations per partition
        """
        partitions       = {}
        partition_counts = {}

        for sample in self.all_samples:
            for base_key, points in sample["data"].items():
                lane = base_key.split("::")[0]

                for p in points:
                    x   = p.get("x")
                    y   = p.get("y", 0.0)

                    # Skip invalid points
                    if x is None or not isinstance(x, (int, float)) or y <= 0.0:
                        continue

                    part_key = self._make_partition_key(base_key, lane, p)

                    if part_key not in partitions:
                        partitions[part_key] = {
                            "x":            [],
                            "y":            [],
                            "group_sizes":  [],
                            "fused_flags":  [],
                            "variables":    [],
                        }
                        partition_counts[part_key] = 0

                    partitions[part_key]["x"].append(float(x))
                    partitions[part_key]["y"].append(float(y))
                    partitions[part_key]["group_sizes"].append(p.get("group_size", -1))
                    partitions[part_key]["fused_flags"].append(bool(p.get("is_fused", False)))

                    # Determine variable type from point metadata
                    variable = "flops" if p.get("x_flops") is not None else "bytes"
                    partitions[part_key]["variables"].append(variable)

                    partition_counts[part_key] += 1

        # ── Collapse metadata lists to scalar summaries ───────────────
        for part_key, data in partitions.items():
            data["variable"]   = self._modal(data["variables"]) or "bytes"
            data["group_size"] = self._modal(
                [g for g in data["group_sizes"] if g > 0], default=-1
            )
            data["any_fused"]  = any(data["fused_flags"])

            # Pass type and dtype/alg_type are encoded in the key itself
            parts = self._parse_partition_key(part_key)
            data["pass_type"] = parts.get("pass_type", "unknown")
            data["dtype"]     = parts.get("dtype",     "unknown")
            data["alg_type"]  = parts.get("alg_type",  "unknown")

        return partitions, partition_counts


    def _merge_degenerate_partitions(self, partitions, partition_counts):
        """
        For any comm partition with < MIN_UNIQUE_X distinct x values,
        merge it into a coarser partition that drops alg_type from the key.
        BUBBLE and COMPUTE partitions are never merged.
        """
        MIN_UNIQUE_X = 3
        merged        = {}
        merged_counts = {}

        for part_key, data in sorted(partitions.items()):
            lane          = part_key.split("::")[0]
            unique_x      = len(set(data["x"]))
            is_degenerate = unique_x < MIN_UNIQUE_X

            if lane in ("BUBBLE", "COMPUTE") or not is_degenerate:
                merged[part_key]        = data
                merged_counts[part_key] = partition_counts[part_key]
                continue

            fallback_key = self._make_fallback_key(part_key)

            if fallback_key not in merged:
                merged[fallback_key] = {
                    "x":           [],
                    "y":           [],
                    "group_sizes": [],
                    "fused_flags": [],
                    "variables":   [],
                }
                merged_counts[fallback_key] = 0

            merged[fallback_key]["x"].extend(data["x"])
            merged[fallback_key]["y"].extend(data["y"])
            merged[fallback_key]["group_sizes"].extend(data["group_sizes"])
            merged[fallback_key]["fused_flags"].extend(data["fused_flags"])
            merged[fallback_key]["variables"].extend(data["variables"])
            merged_counts[fallback_key] += partition_counts[part_key]

        # Re-derive scalar metadata for merged buckets that lack it
        for part_key, data in merged.items():
            if "variable" not in data:
                data["variable"]   = self._modal(data.get("variables", [])) or "bytes"
            if "group_size" not in data:
                data["group_size"] = self._modal(
                    [g for g in data.get("group_sizes", []) if g > 0], default=-1
                )
            if "any_fused" not in data:
                data["any_fused"]  = any(data.get("fused_flags", []))
            parts = self._parse_partition_key(part_key)
            data["pass_type"] = parts.get("pass_type", "unknown")
            data["dtype"]     = parts.get("dtype",     "mixed")
            data["alg_type"]  = parts.get("alg_type",  "mixed")

        return merged, merged_counts

    @staticmethod
    def _make_fallback_key(part_key):
        """
        Drop alg_type from a comm partition key, keeping dtype.

        "EP_COMM::AlltoAllV[BFP16|MESH-RING-NHR]"  →  "EP_COMM::AlltoAllV[BFP16]"
        "DP_COMM::AllGather[FP32|RING-NHR-NHR]"    →  "DP_COMM::AllGather[FP32]"
        "COMPUTE::MatMul[forward]"                  →  "COMPUTE::MatMul[forward]"
        "BUBBLE::Wait"                              →  "BUBBLE::Wait"
        """
        if "[" not in part_key:
            return part_key
        base   = part_key.split("[", 1)[0]
        suffix = part_key.split("[", 1)[1].rstrip("]")
        if "|" not in suffix:
            return part_key          # COMPUTE[pass_type] — leave unchanged
        dtype = suffix.split("|", 1)[0]
        return f"{base}[{dtype}]"
  
    @staticmethod
    def _make_partition_key(base_key, lane, point):
        """
        Build the sub-partitioned key for a data point.

        BUBBLE               → base_key unchanged
        COMPUTE              → base_key[pass_type]
        communication lanes  → base_key[dtype|alg_type]
        """
        if lane == "BUBBLE":
            return base_key

        if lane == "COMPUTE":
            pass_type = point.get("pass_type", "unknown") or "unknown"
            return f"{base_key}[{pass_type}]"

        # Communication lane
        dtype    = point.get("dtype",    "unknown") or "unknown"
        alg_type = point.get("alg_type", "unknown") or "unknown"
        return f"{base_key}[{dtype}|{alg_type}]"

    @staticmethod
    def _parse_partition_key(part_key):
        """
        Extract metadata fields encoded in a partition key.

        Examples
        ────────
        "EP_COMM::AlltoAllV[BFP16|MESH-RING-NHR]"
            → {dtype: "BFP16", alg_type: "MESH-RING-NHR"}
        "COMPUTE::MatMul[forward]"
            → {pass_type: "forward"}
        "BUBBLE::Wait"
            → {}
        """
        result = {}
        if "[" not in part_key:
            return result

        suffix = part_key.split("[", 1)[1].rstrip("]")
        lane   = part_key.split("::")[0]

        if lane == "COMPUTE":
            result["pass_type"] = suffix
        else:
            parts = suffix.split("|", 1)
            if len(parts) == 2:
                result["dtype"]    = parts[0]
                result["alg_type"] = parts[1]

        return result

    # ------------------------------------------------------------------ #
    #  Outlier filtering                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _iqr_filter(x_vals, y_vals):
        """
        Remove points whose y value lies above Q3 + k×IQR.

        We only filter the upper tail because the lower tail (fast events)
        represents the true transfer time. Upper-tail outliers are pipeline
        stall times that contaminate PP_COMM Receive durations.

        Returns
        -------
        x_clean    : ndarray   filtered x values
        y_clean    : ndarray   filtered y values
        n_removed  : int       number of points removed
        """
        q1, q3     = np.percentile(y_vals, [25, 75])
        iqr        = q3 - q1
        upper_fence = q3 + _IQR_MULTIPLIER * iqr

        mask      = y_vals <= upper_fence
        n_removed = int(np.sum(~mask))
        return x_vals[mask], y_vals[mask], n_removed

    # ------------------------------------------------------------------ #
    #  Hockney fitting                                                     #
    # ------------------------------------------------------------------ #

    def _fit_hockney(self, lane, key, x_vals, y_vals, any_fused=False):
        """
        Fit T(n) = α + n/β via OLS and return (alpha, beta, r2,
        low_confidence, confidence_reason).

        Special cases
        ─────────────
        BUBBLE lane
            Latency-only: α = mean(y), β = ∞, r2 = 1.0.

        Fewer than 2 points after filtering
            Cannot fit at all: α = mean(y), β = ∞, low_confidence = True.

        Single unique x value
            OLS is degenerate: moment-based fallback, low_confidence = True.

        Non-positive OLS slope
            Physically impossible (more work → less time).
            Degrade to latency-only (β = ∞), low_confidence = True.

        is_fused = True
            IR tensor sizes are unreliable due to compiler fusion.
            Fit proceeds but low_confidence is set.
        """
        low_confidence   = False
        confidence_reason = ""

        # ── Propagate fused flag ──────────────────────────────────────
        if any_fused:
            low_confidence    = True
            confidence_reason = "IR sizes unreliable due to compiler fusion"

        # ── BUBBLE: latency-only ──────────────────────────────────────
        if lane == "BUBBLE":
            alpha = float(np.mean(y_vals)) if len(y_vals) > 0 else 0.0
            return max(0.0, alpha), float("inf"), 1.0, low_confidence, confidence_reason

        # ── Not enough points ─────────────────────────────────────────
        if len(x_vals) < 2:
            alpha = float(np.mean(y_vals)) if len(y_vals) > 0 else 0.0
            reason = "fewer than 2 points after outlier filtering"
            return (
                max(0.0, alpha), float("inf"), 0.0,
                True, reason
            )

        # ── Single unique x ───────────────────────────────────────────
        if len(np.unique(x_vals)) < 2:
            print(
                f"[interpreter] WARNING: {key} — only one unique x value "
                f"({x_vals[0]:.3g})"
            )
            alpha, beta, r2 = self._moment_fallback(x_vals, y_vals)
            reason = f"single unique x={x_vals[0]:.3g}; moment-based estimate"
            return alpha, beta, r2, True, reason if not low_confidence else confidence_reason

        # ── OLS ───────────────────────────────────────────────────────
        slope, intercept, r_value, _, _ = stats.linregress(x_vals, y_vals)
        r2    = float(r_value ** 2)
        alpha = max(0.0, float(intercept))

        if slope <= 1e-12:
            print(
                f"[interpreter] WARNING: {key} — slope={slope:.3e} ≤ 0 "
                f"(noisy data over narrow range). Latency-only fallback."
            )
            reason = f"non-positive OLS slope ({slope:.3e}); latency-only"
            return (
                alpha, float("inf"), r2,
                True, reason if not low_confidence else confidence_reason
            )

        beta = float(1.0 / slope)
        return alpha, beta, r2, low_confidence, confidence_reason

    @staticmethod
    def _moment_fallback(x_vals, y_vals):
        """
        Moment-based Hockney estimate when OLS is degenerate.

        Splits mean duration as:  α ≈ 10 %,  n/β ≈ 90 %
        Returns (alpha, beta, r2=0.5).
        """
        mean_y  = float(np.mean(y_vals)) if len(y_vals) > 0 else 0.0
        mean_x  = float(np.mean(x_vals)) if len(x_vals) > 0 else 1.0
        alpha   = mean_y * 0.10
        bw_time = mean_y * 0.90
        beta    = (
            float(mean_x / bw_time)
            if bw_time > 1e-12 and mean_x > 0
            else float("inf")
        )
        return max(0.0, alpha), beta, 0.5

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _modal(values, default=None):
        """Return the most common value in a list, or default if empty."""
        if not values:
            return default
        try:
            return stat_mode(values)
        except Exception:
            # If all values are equally common, return the first one
            return values[0] if values else default

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #

    def _print_summary(self):
        """Print a per-lane summary grouped by base primitive."""
        # Group by lane for top-level summary
        lane_entries = defaultdict(list)
        for key, f in self.formula.items():
            lane = key.split("::")[0]
            lane_entries[lane].append((key, f))

        print()
        for lane in sorted(lane_entries):
            entries = lane_entries[lane]
            avg_r2  = sum(f["r2"] for _, f in entries) / len(entries)
            n_low   = sum(1 for _, f in entries if f["low_confidence"])
            #print(
            #    f"[interpreter] {lane:<14}  "
            #    f"{len(entries):>3} partition(s)   "
            #    f"avg R²={avg_r2:.4f}   "
            #    f"low-confidence={n_low}/{len(entries)}"
            #)
            for key, f in sorted(entries, key=lambda kv: -kv[1]["count_per_step"]):
                # Strip lane prefix for display
                display = key.split("::", 1)[1]
                bw = f"{f['beta']:,.1f} {f['variable']}/µs" if f["beta"] != float("inf") else "∞"
                gs        = f["group_size"]
                gs_str    = f""
                print(
                        f"{display}: "
                        f"α : {f['alpha']}  "
                        f"β : {bw}  "
                   # f"count/step={f['count_per_step']:>7.1f}  "
                   f"R² : {f['r2']:.4f}  "
                    #f"{gs_str}"
                )
                #if f["low_confidence"] and f["confidence_reason"]:
                #    print(f"               {'':48}  ↳ {f['confidence_reason']}")
        print()
