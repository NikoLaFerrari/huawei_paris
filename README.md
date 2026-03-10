# SAPP: Symbolic and Automatic Parallel Planner

## 1.Overview
SAPP is a solution aiming to configure parallelism for LLM training.
It contains several modules responsible for different set of parallelism features organized as follows:

![SAPP Overview](figures/sapp_overview.png)

### I. ND: parallelizing N Dimensions with symbolic estimation

ND provides a degree to *N* parallelism dimensions, also known as N-D. 
The analytical nature of this module enables:
- Exhaustive exploration $\to$ All configuration are estimated
- Very fast exploration $\to$ Seconds
- No reliance on the execution cluster $\to$ Only CPU
More details can be found in the Readme ```paradise/README.md```.

### II. PPB: Pipeline Parallelism Balancing
Naive pipeline parallelism orchestration lead to imbalances in computation because of unforeseen different loads of different layers, including beginning and final computation. It also leads to imbalanced memory footprint caused by the chosen pipeline schedule. To achieve optimal performance, it is essential to balance both computational and memory loads simultaneously, including recomputation.
In such cases, the SAPP pipeline load balancing tool is required to automatically generate the optimal strategy.
More details on the software can be found in the paper [ [1] ](#BMPipe) or the Readme ```pipeline_balance/README.md```.

### III. MemEst: Memory Estimation
This module estimates memory usage in large language model (LLM) training under various parallelism strategies. It predicts peak memory consumption (both static and dynamic), provides detailed memory insights, and can visualize results with plots.
More details can be found in ```memory_estimation/README.md```.


## 2. Utilization

Here is a case we had on DeepSeekV3 with 15 layers on 128 devices, 2048 of global batch size for device A3 (910C). Following options are meant for a given balancing, swap optimizer, and to display top 10.

```bash
cd paradise
python run_paradise.py -y yamls/ds/DS_128p_15L.yaml -l DP MP EP PP MB -d 128 -b 2048 -A3 -mppb -swap_os -t 10
```

Which outputs the following plot in ```output/results.pdf```:

![ex_result](figures/nd_result.png)

Now we have the best ND returned: 32 DP, 2 MP, 32 EP, 2 PP, 64 MB.
We set this configuration in the yaml (or a copy) such as:

```yaml
parallel_config:
  data_parallel: &dp 32
  model_parallel: 2
  pipeline_stage: 2
  expert_parallel: 64
  micro_batch_num: 64
```

We run memory estimation on this yaml to get a layer file for PPB.
Because this uses swap optimizer, it needs to be specified in ```memory_estimation/configs_eval/default.yaml``` such as

```yaml
...
passes :
  swap_optimizer : True
  ...
...
```

Then we run memory estimation and generate the layer file
```bash
cd ../memory_estimation
python estimate_v2.py ../paradise/yamls/ds/DS_128p_15L.yaml --ppb > layer_file.json
```

Then the content of layer_file.json needs to be edited a bit.
- Select recompute is not used in that case so every field ```'memory_select_rec'``` needs to be removed. 
- And a time for each layer type needs to be added. It is best if this time comes from a profiling of the current model with the ND found but otherwise, it can be expert guessed or taken from a similar enough case. Here, we will put time 0 for HEAD, 20 for dense (BODY_2), 35 for MoE (BODY_3) and 10 for TAIL.

After edition, the layer file should look like this:

```json
{
    "layers_description": [
        {
            "memory_activation": 857,
            "memory_parameter": 631,
            "memory_recompute": 56,
            "model_name": "deepseekV3",
            "name": "BODY_2",
            "nb_layer": 3,
            "time": 20,
            "type": "BODY"
        },{
            "memory_activation": 1111,
            "memory_parameter": 2924,
            "memory_recompute": 56,
            "model_name": "deepseekV3",
            "name": "BODY_3",
            "nb_layer": 13,
            "time": 35,
            "type": "BODY"
        },{
            "memory_parameter": 275,
            "model_name": "deepseekV3",
            "name": "HEAD",
            "nb_layer": 1,
            "time": 0,
            "type": "HEAD"
        },{
            "memory_parameter": 9629,
            "model_name": "deepseekV3",
            "name": "TAIL",
            "nb_layer": 1,
            "time": 10,
            "type": "TAIL"
        }
    ]
}
```

One may notice that this 15 layer DeepSeek here has 3 Dense + 13 MoE. That is because the MTP layer at the end is part copy of the last MoE layer, so we consider the MTP with 1 additional MoE layer and the rest in the Tail.

Now that we have ND and the layer file, we can run pipeline balancing with parameters given by ND: 2 stages (PP), 64 MB as well as the dual pipe schedule (from the yaml configuration) which requires 2 interleave.

```bash
mv layer_file.json ../pipeline_balance/layers/ # Move layer file to the suitable directory 
cd ../pipeline_balance                         # Go to pipeline balancing directory
python run_pipeline_balance.py -m layer_file -s 2 -mb 64 -sc dual -i 2
```

which outputs this visualization of the pipeline execution and memory:

![plot](figures/ppb_result.png)

and this log:

``` bash
...
To put in yaml configuration:
        offset: [[3, 0], [0, 0]]
        recompute: [[0, 0], [0, 0]]
        select_recompute: [[0, 0], [0, 0]]
        select_comm_recompute: [[0, 0], [0, 0]]
2025-12-17 15:25:50,593 - pipeline_balance - OUTPUT - layer_num = 13
2025-12-17 15:25:50,593 - pipeline_balance - OUTPUT - layer-to-stage assignment baseline is
        [[3, 3], [3, 3]]
2025-12-17 15:25:50,594 - pipeline_balance - OUTPUT -
To put in yaml configuration:
        offset: [[-1, 1], [0, 1]]
        recompute: [[0, 0], [0, 0]]
        select_recompute: [[0, 0], [0, 0]]
        select_comm_recompute: [[0, 0], [0, 0]]
...
```
Which means no recomputation was needed and the global offset is the element-wise addition between the offsets for both dense layers & moe layers. which is 
```math
\newcommand{\offset}{\mathit{offset}}
\begin{array}{lcl}
    \offset &=& \offset_{dense} + \offset_{moe} \\
            &=& [[3, 0], [0, 0]] + [[-1, 1], [0, 1]]\\
            &=& [[2, 1], [0, 1]] \\
    \offset &\equiv& [[1, 0], [-1, 0]]
\end{array}
```

Here is a comparison with a baseline configuration quickly found by an expert.

|          | DP | MP | EP | PP | MB  | Offset               | Recompute | Step time | MFU |
| -------- | -- | -- | -- | -- | --- | -------------------- | --------- | --------- | --- |
| Baseline | 8  |  4 | 32 |  4 | 256 | ```[[0,0], [0,0]]``` | False     | 44.396s   | 29% |
| SAPP     | 32 |  2 | 64 |  2 |  64 | ```[[1,0],[-1,0]]``` | False     | 36.268s   | 36% |

It demonstrates a speedup of 18%.


## 3. Structure

```bash
toolkits/
├── figures/                        # Directory containing figures used in this doc
├── memory_estimation/              # MemEst for symbolic memory estimation
├── paradise/                       # Paradise for ND
├── perf_estimation/                # Performance estimation for ND
├── pipeline_balance/               # PPB
└── README.md                       # This file
```

## Future work

- [ ] New Frameworks, beside MindSpore
- [ ] New models, including multi-modal
- [ ] New Hardware
- [ ] Add an Orchestrator to
    - Streamline the connection between ND and PPB
    - Centralize inputs and outputs

## Other Projects
### A. D-REC: Double RECursion for tensor parallelism
This module aims at partitioning each tensor of the DNN while minimizing the entailed communication overhead.
Its recursion enables a logarithmic complexity in the number of devices involved. Moreover, it focuses on important (or complex) operators while simple operators' partitioning are inferred witch sharding propagation.
This lightweight design enables an exploration of the whole graph for the given number of devices in seconds while ensuring a low communication cost.
More details can be found in the papers [ [2] ](#DRec) and [ [3] ](#Rapid). This module is currently implemented in MindSpore in <https://gitcode.com/mindspore/mindspore/tree/master/mindspore/ccsrc/frontend/parallel/auto_parallel/rec_core>.


## References

<a id="BMPipe" name="BMPipe">[1]</a> Wang, Ruiwen, et al. "BMPipe: bubble-memory co-optimization strategy planner for very-large DNN training." 2025 IEEE International Conference on Cluster Computing (CLUSTER). IEEE, 2025.

<a id="DRec" name="DRec">[2]</a> Wang, Haoran, et al. "Efficient and systematic partitioning of large and deep neural networks for parallelization." European Conference on Parallel Processing. Cham: Springer International Publishing, 2021.

<a id="Rapid" name="Rapid">[3]</a> Tachon, Thibaut, Haoran Wang, and Chong Li. "RAPID: A Rapid Automatic Parallelizer for Immense Deep Neural Networks." 2024 IEEE International Conference on Cluster Computing Workshops (CLUSTER Workshops). IEEE, 2024.