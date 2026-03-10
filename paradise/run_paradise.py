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
"""run parallelization"""

import argparse
import os

from memory_estimation.size import Memory
from paradise.logger import logger, set_verbose_level
import paradise.parallelize as Par
import paradise.dimensions as Dim
import paradise.common.hardware as Hard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python run_paradise.py",
        description=("Provides a degree to *N* parallelism dimensions"),
        epilog="",
    )

    parser.add_argument(
        "-y",
        "--yaml_config",
        type=str,
        required=True,
        help="Path to yaml configuration file",
    )
    parser.add_argument(
        "-d",
        "--devices",
        type=int,
        default=None,
        help="Number of devices. Takes yaml value if unspecified",
    )
    parser.add_argument(
        "-b",
        "--global_batch_size",
        type=int,
        default=None,
        help="Global batch size. Takes yaml value if unspecified",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model Name to use. Takes yaml value if unspecified",
    )
    # parser.add_argument(
    #     "-g",
    #     "--generate_yaml_in",
    #     type=str,
    #     default=None,
    #     help="Generate all fitting yaml configurations in the given folder",
    # )
    # parser.add_argument(
    #     "-c",
    #     "--csv",
    #     type=str,
    #     default=None,
    #     help="Computes correlation coefficient from csv results file",
    # )
    parser.add_argument(
        "-l",
        "--dimensions",
        nargs="*",
        type=str,
        default=None,
        help="list of varying (output) dimensions",
    )
    # parser.add_argument(
    #     "-j",
    #     "--threads_num",
    #     type=int,
    #     default=None,
    #     help="Number of threads for the space generation",
    # )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=2,
        help="Level of verbosity in range [0,6], "
        "0 being no output and 6 being debug level output. "
        "Plot and debug csv are generated from 2",
    )
    # parser.add_argument(
    #     "-k",
    #     "--ppb_k",
    #     type=int,
    #     default=None,
    #     help="choose configuration number k for ppb",
    # )
    parser.add_argument(
        "-A",
        "--device_type",
        type=int,
        default=2,
        help="choose device type between A2 or A3",
    )
    parser.add_argument(
        "-swap_os",
        "--swap_opt_state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Activate swap optimiezr state",
    )
    # parser.add_argument(
    #     "-lm",
    #     "--less_memory",
    #     action=argparse.BooleanOptionalAction,
    #     default=False,
    #     help="Activate less memory schedule",
    # )
    parser.add_argument(
        "-mppb",
        "-–manual_pipeline_balance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Takes offset and recompute from yaml",
    )
    parser.add_argument(
        "-t",
        "--top_config_number",
        type=int,
        default=None,
        help="Number of top configs to print & plot",
    )
    parser.add_argument(
        "-mem",
        "--mem_for_ppb",
        type=str,
        default="0GB",
        help="Memory to reserve for pipeline balancing. "
        "Will be decreased from the memory budget allowed by ND (default 0GB)",
    )
    parser.add_argument(
        "-c",
        "--cache_file",
        type=str,
        default=None,
        help="Cache file with ratios to recalibrate ND scores. "
        "Will be defaulted to 'None'."
    )

    args = parser.parse_args()

    if args.cache_file is not None:
        if not os.path.exists(args.cache_file):
            logger.error(
                    f"cache file not found:"
                    f" {args.cache_file}"
                     "\nProceeding without cache file...")
            args.cache_file = None

    set_verbose_level(args.verbosity)
    dims = Dim.get_dims(args.dimensions)
    YAML_FOLDER = None  #args.generate_yaml_in
    machine = Hard.Machine(args.devices, args.device_type)
    paradise = Par.Parallelize(
        args.yaml_config,
        machine,
        args.global_batch_size,
        dims,
        swap_os=args.swap_opt_state,
        mppb=args.mppb,
        model=args.model,
        # model="Telecom",  # args.model ====ONLY FOR XINYU BRANCH====
        mem_for_ppb=Memory.from_string(args.mem_for_ppb.strip()),
        # vpp_less_mem=args.less_memory,
    )

    if YAML_FOLDER and not os.path.exists(YAML_FOLDER):
        os.makedirs(YAML_FOLDER)

    space = paradise.run_generation_to_ordering(
        YAML_FOLDER,
        threads_num=None,  # args.threads_num
        top_num=args.top_config_number,
        cache_file=args.cache_file,
    )
