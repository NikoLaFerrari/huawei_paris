# Regression:
This framework provides an automated way of extracting real-hardware performance characteristics from trace files and their corresponding IR Graphs, and using them to calibrate ```perf_estimation```'s scoring. By analyzing the relationship between work volume (FLOP) and execution time, it allows for performance prediction on similar configs as the aforementioned trace files.

## Framework:
The framework follows the follwing pipeline:
1. ```Extractor```: Parses trace files and IR graphs using bench_tools methods to categorize compute and communication events.
2. ```Interpreter```: Performs statistical regression on extracted data points to find coefficients (alpha & beta)
3. ```Predictor```: Extrapolates the learnt formulae to target parallel dimensions.
4. ```Handler```: Manages th end-to-end flow, IO and caching.

## File Breakdown:
1. ```interface.py```:
    - Contains ```Handler``` class.
    - Takes input:
      - ```trace_paths```: path(s) to trace files for extraction.
      - ```config_paths```: path(s) to the respective config files.
      - ```graph_paths```: path(s) to the respective IR Graph files.
      - ```input_dims```: config/dimensions to predict the execution time for
    - Handles intergration among ```Paradise - Extractor - Interpreter - Predictor```.
    - Runs the Paradise engine internally to obtain ND's theoretical classification. Also allows for obtaining optimal top-k strategies from ND and then predicting the total time each strategy would take to execute using ```Predictor```.
    - Obtains extracted data from input trace files and IR graphs using ```Extractor```. Extracted data is sent to ```Interpreter``` to obtain per-bucket primitive formulae. Then ```Predictor``` is called with the target config and formulae to predict the executaion time when running the target config.
    - Compares bench_tools classification against ND's classification to obtain ratios.
    - Stores calibrated ratios in ```cache/<model_name>.json``` for ```perf_estimation``` to use.


2. ```extractor.py```:
    - Contains ```Extractor``` class.
    - Input:
      - ```trace_paths```: path(s) to trace files for extraction.
      - ```config_paths```: path(s) to the respective config files.
      - ```graph_paths```: path(s) to the respective IR Graph files.
    - Outputs:
      - ```all_samples```: aggregated classification dictionary (explained in the next points).
    - Goes through a trace files and its IR Graph to extract primitives using ```bench_tools.prof``` and ```bench_tools.ms_trace```. 
    - Then proceeds to classify the primitives into ```bench_tools``` buckets using ```bench_tools.results.comm_classification```. 
    - After, a mapping occurs to convert ```bench_tools``` buckets into ```regression/ND``` buckets. The final classification would look as follows:
    ```
    all_samples = {
     'COMPUTE'
        {<comp_primitive_1> {comp_primitive_data}, ...}, 
     'DP_COMM':
        {dp_primitive_1 {dp_primtive_data}, ...}, 
     'MP_COMM': 
        {mp_primitive_1 {mp_primitve_data}, ...},
     'EP_COMM':
         {ep_primtive_1 {ep_primtive_data}, ...},
     ... # and so on with other buckets 
    }
    ```
    - If multiple trace files and their IR Graphs are inputted, points 2-4 are repeated for each pair and an aggregated classification dictionary is created.
    - This aggregated dictionary is returned.

3. ```interpreter.py```:
    - Contains ```Interpreter``` class.
    - Input:
      - ```all_samples```: output of ```Extractor```, the aggregated classification dictionary.
    - Outputs:
      - ```formulae```: formulae generated from regression on ```all_samples```.
    - For each per-bucket primitive, the ```all_samples``` data is converted to an "x-y" dictionary as follows:
    ```
    bucket_points = {
	<bucket>::<primtive1_name>: {x: size, y: duration}
	<bucket>::<primtive2_name>: {x: size, y: duration}
        ...
    }
    ``` 
    - Then for each ```<bucket>::<primtive>```, we perform regression to obtain the slope, intercept, variable to scale over, number of said primitive per step and confidence(R^2 value) using ```scipy.stats.linregress```. These values make a formula.
    - These formulae gets added into a dictionary containing all the formulae: ```formula``` .
    - ```formula``` is returned.
    - **Note**: If the trace files were trained over the same ```device_number``` then the ```x``` in ```bucket_points``` would be equal and hence no regression can be done. Hence the requirement at least two 3-tuples (trace, config, graph) trained over different ```device_number```. 
   
5. ```predictor.py```:
    - Contains ```Predictor``` class.
    - Input:
        - ```model```: name changed from ```formula```.
        - ```coeffs```: [Experimental] for optimal coefficients for mp, dp, ep scaling.
    - Ouput:
	- ```total_time```: predicted total time for target config
	- ```bucket_times```: predict bucket times for target config
    - For each primitive in a bucket, the respective formula is extrapolated to the scaling variable from target config.
    - This process is repeated for all the primitives in that bucket, then summed up to obtain ```bucket_times``` - the predicted time each bucket will take.
    - The sum of all the values in ```bucket_times``` becomes ```total_time```.
    - ```total_time``` is returned.

## Structure:
```
regression/
├── README.md
├── bench_tools
│   ├── ir
│   │   ├── graph.py
│   │   └── parser.py
│   ├── ms_trace.py
│   ├── prof.py
│   ├── results
│   │   ├── bench_result.py
│   │   └── comm_classification.py
│   └── utils
│       ├── base_utils.py
│       ├── ir_utils.py
│       └── results_utils.py
├── cache				# folder containing cache files
│   └── deepseekV3.json			# examples cache file for deepseekV3.py
├── extractor.py			# handles the extraction process, Extractor
├── interface.py			# handles inter- and intra-integration, Handler
├── interpreter.py			# handles analysing the extracted data, Interpreter
├── plot_theta.py			# for plotting graphs comparing regression prediction vs ND prediction
├── predictor.py			# handles prediction for target configs, Predictor
└── run_regression.py			# CLI for regression
```

## Help:
``` python run_regression.py -h ```

## Run:
``` 
python run_regreesion.py \
	-t [path to trace files] \
	-g [path to graph files] \
	-c [path to config files] \
	-i input dimensions (target config)
```
However, since ```run_paradise.py``` is not complete due to the incomplete ```predictor.py```, we propose that ```regression``` be run as follows:
```
from interface import Handler

input_dims = {} 	# desired prediction dims

trace_paths = [] 	# paths to trace files

graph_paths = []	# paths to IR Graphs

config_paths = [] 	# paths to trace configs

handler = Handler(trace_paths, config_paths, graph_paths, input_dims)
handler.run_calibration()
```
One can copy-paste this into a python file and run it to run ```regression``` until ```run_regression.py``` has been completed.

Note: graph_paths[i] and config_paths[i] should be the trace_paths[i] files IR Graph and trace config files respectively.
