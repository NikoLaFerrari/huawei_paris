import argparse
import yaml
import json
import os
from interface import Handler
from predictor import Predictor


def main():
    parser = argparse.ArgumentParser(description="regression CLI.")

    parser.add_argument(
        "-t", "--traces", nargs="+", required=True,
        help="List of paths to trace_view.json files."
    )
    parser.add_argument(
        "-c", "--configs", nargs="+", required=True,
        help="List of paths to the trace yaml configs."
    )
    parser.add_argument(
        "-g", "--graphs", nargs="+", required=True,
        help="List of paths to the graph directories."
    )
    parser.add_argument(
        "-i", "--target_config", required=True,
        help="Path to the yaml config you want to PREDICT for."
    )
    parser.add_argument(
        "-l", "--layers", nargs="+",
        default=['PP', 'MP', 'DP', 'EP', 'MB', 'vpp'],
        help="List of dimensions to consider during prediction (e.g., MP PP EP CP MB)"
    )

    args = parser.parse_args()

    with open(args.target_config, 'r') as f:
        target_raw = yaml.safe_load(f)

    # Extract target dims via a throwaway Handler (no paths, no config).
    # Handler.__init__ requires 5 args; pass None for input_config and {}
    # for input_dims since we are only calling extract_trace_dims here.
    dummy = Handler([], [], [], None, {})
    target_dims, _, _ = dummy.extract_trace_dims(target_raw)

    device_count = target_raw.get('parallel', {}).get('device_num', 128)
    gbs          = target_raw.get('model', {}).get('model_config', {}).get('batch_size', 128)

    input_dims = {
        'dims':         args.layers,
        'device_count': device_count,
        'gbs':          gbs,
    }

    handler = Handler(
        args.traces,
        args.configs,
        args.graphs,
        args.target_config,
        input_dims,
    )

    handler.run_calibration()

    # Final standalone prediction for the specified target config.
    predictor = Predictor(
        handler.formula,
        handler.meta['trace_dims_list'][0],
        coeffs=handler.optimized_coeffs,
    )
    total_time, breakdown = predictor.predict_with_breakdown(target_dims)
    bucket_times          = breakdown["bucket_times"]

    print(f"\nPREDICTING: {target_dims}")
    print(f"Predicted Time: {total_time:.2f} µs  ({total_time / 1e6:.4f} s)")

    raw_total = sum(bucket_times.values())
    pct = {
        lane: (val / raw_total) * 100 if raw_total > 0 else 0.0
        for lane, val in bucket_times.items()
    }
    print("\nPer-lane breakdown (% of raw lane totals):")
    print(json.dumps(pct, indent=2))

    print("\nPer-primitive top-20 by predicted time:")
    top20 = sorted(
        breakdown["primitive_times"].items(),
        key=lambda kv: kv[1],
        reverse=True,
    )[:20]
    for key, t in top20:
        print(f"  {key:<60} {t:>12.1f} µs")


if __name__ == "__main__":
    main()
