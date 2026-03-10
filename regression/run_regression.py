import argparse
import yaml
import json
import os
from interface import Handler
from predictor import Predictor

def main():
    parser = argparse.ArgumentParser(description="regression CLI.")

    parser.add_argument("-t", "--traces", nargs="+", required=True, help="List of paths to trace_view.json files.")
    parser.add_argument("-c", "--configs", nargs="+", required=True, help="List of paths to the trace yaml configs.")
    parser.add_argument("-g", "--graphs", nargs="+", required=True, help="List of paths to the graph directories.")
    parser.add_argument("-i", "--target_config", required=True, help="Path to the yaml config you want to PREDICT for.")
    parser.add_argument("-l", "--layers", nargs="+", default=['PP', 'MP', 'DP', 'EP', 'MB', 'vpp'],
                        help="List of dimensions to consider during prediction (e.g., MP PP EP CP MB)")

    args = parser.parse_args()

    with open(args.target_config, 'r') as f:
        target_raw = yaml.safe_load(f)
    
    dummy = Handler([], [], [], {})
    target_dims = dummy.extract_trace_dims(target_raw)
    
    device_count = target_raw.get('parallel', {}).get('device_num', 128)
    gbs = target_raw.get('model', {}).get('model_config', {}).get('batch_size', 128)

    input_dims = {
        'dims': args.layers,
        'device_count': device_count,
        'gbs': gbs
    }

    handler = Handler(args.traces, args.configs, args.graphs, input_dims)

    handler.run_calibration()

    predictor = Predictor(handler.formula, handler.trace_dims_list[0], coeffs=handler.optimized_coeffs)
    total_time, breakdown = predictor.predict_with_breakdown(target_dims)

    print(f"\nPREDICTING: {target_dims}")
    print(f"Predicted Time: {total_time:.2f} ms")
    print(json.dumps(breakdown, indent=2))
    
    pct = {k: (v/sum(breakdown.values()))*100 for k, v in breakdown.items() if sum(breakdown.values()) > 0}
    print("\nPercentage:")
    print(json.dumps(pct, indent=2))

if __name__ == "__main__":
    main()
