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

"""primary util methods"""

import copy
import glob
import hashlib
import json
import os
import subprocess
import sys
from collections import OrderedDict

import yaml
from loguru import logger

def set_up_logger(logger: any) -> None:
    """Set up custom logger."""
    logger.remove()
    logger.level("DEBUG", color="<blue>")
    logger.level("INFO", color="<magenta>")
    logger.level("SUCCESS", color="<green>")
    logger.add(
        sys.stdout,
        level="DEBUG",
        format="[<level>{level}</level>][<level>{function}</level>] {message}",
    )


set_up_logger(logger)


def run_commands(commands: any, env: dict = None) -> None:
    """Run shell commands through python, in batch."""
    if isinstance(commands, str):
        commands = [commands]
    for command in commands:
        process = subprocess.Popen(command, shell=True, env=env)
        process.wait()


def get_rank_dirs(path):
    """ return sorted rank dir names inside path """
    rank_path_pattern = os.path.join(path, "rank_*")
    rank_dirs = glob.glob(rank_path_pattern)
    rank_dirs = [d.split("/")[-1] for d in rank_dirs]
    rank_dirs = sorted(rank_dirs, key=lambda d: int(d.split("_")[-1]))
    return rank_dirs


def get_profile_path(bench_dir: str, rank=None):
    """ 
    return the resolved profiling dir path for bench run
    None if not found    
    """
    base_profile_dir = os.path.join(bench_dir, "output/profile")
    rank_dirs = get_rank_dirs(base_profile_dir)
    if rank is None:
        if len(rank_dirs) <= rank:
            raise ValueError(
                f"Asked for local rank {rank} while only "
                f"{len(rank_dirs)} local ranks are available "
                f"in profile dir {base_profile_dir}"
            )
        rank_dir = rank_dirs[0]
    else:
        rank_dir = f"rank_{rank}"
    base_profile_dir_pattern = os.path.join(
        base_profile_dir, rank_dir, "*_ascend_ms"
    )
    matching_dir_list = glob.glob(base_profile_dir_pattern)
    if len(matching_dir_list) == 0:
        return None
    if len(matching_dir_list) > 1:
        logger.info(
            f"More than 1 profiling directory "
            f"detected ({len(matching_dir_list)}). Taking first one"
        )
    profile_dir = matching_dir_list[0]

    return profile_dir


def get_backend_run(bench_cfg: dict) -> str:
    """Get backend type where the benchmark will run."""
    if (
        "cloud_out_path" in bench_cfg
        and bench_cfg["cloud_out_path"] is not None
        and bench_cfg["cloud_out_path"] != ""
    ):
        return "cloud"
    if (
        "cluster_rank" in bench_cfg
        and bench_cfg["cluster_rank"] is not None
        and bench_cfg["cluster_rank"] != ""
    ):
        return "cluster"
    return "local"


def get_log_file(bench_cfg, nb_cards):
    """
    return the worker log filename to read 
    based on backend type and card count
    """
    backend_run = get_backend_run(bench_cfg)
    if backend_run in ["cloud", "cluster"]:
        if backend_run == "cloud":
            cur_worker = int(
                os.getenv("MA_CURRENT_INSTANCE_NAME").split("-")[-1]
            )
        else:
            cur_worker = bench_cfg["cluster_rank"]

        log_file_idx = (cur_worker + 1) * nb_cards - 1
        log_file = f"worker_{log_file_idx}.log"
    else:
        log_file = "worker_0.log"

    return log_file


def generate_hash_name(dict_obj: dict) -> str:
    """Generate fix hash name from a dictionnary object."""
    hash_name = hashlib.blake2b(digest_size=6)
    hash_name.update(str(dict_obj).encode())
    return hash_name.hexdigest()


def ordered_yaml_dump(
    data: any,
    stream: any = None,
    yaml_dumper: any = yaml.SafeDumper,
    object_pairs_hook: any = OrderedDict,
    **kwargs: any,
) -> None:
    """Dump Dict to Yaml File in Orderedly.

    This function comes from Mindformers.
    It is used to respect the same Mindformers dictionnary style order.
    """

    class OrderedDumper(yaml_dumper):
        """ OrderedDumper class, not implemented """
        pass

    def _dict_representer(dumper: any, data: any) -> None:
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items()
        )

    OrderedDumper.add_representer(object_pairs_hook, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwargs)


def check_file_exists(file_path: str, is_quit: bool = False) -> bool:
    """Check if a file exsists. If it doesn't exist, quit."""
    if not os.path.isfile(file_path):
        if is_quit:
            logger.error(
                'The file: "{file_path}" doesn\'t exist', file_path=file_path
            )
            sys.exit()
        return False
    return True


def check_dir_exists(dir_path: str) -> None:
    """Check if a directory exsists. If it doesn't exist, quit."""
    if not os.path.isdir(dir_path):
        logger.error(
            'The directory: "{dir_path}" doesn\'t exist', dir_path=dir_path
        )
        sys.exit()


def create_exit_dir(dir_path: str) -> bool:
    """Create a directory to the given path. If it already exist, nothing is done."""
    if os.path.isdir(dir_path) or os.path.isfile(dir_path + ".zip"):
        logger.debug(
            "The directory {dir} exists, will be skipped...", dir=dir_path
        )
        return True
    os.mkdir(dir_path)
    logger.debug("Created directory {dir}", dir=dir_path)
    return False


def open_yaml(file_path: str) -> dict:
    """Load and parse a yaml file."""
    with open(file_path, encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.Loader)


def save_yaml(obj: dict, file_path: str) -> None:
    """Save a dictionnary object to YAML file."""
    with open(file_path, "w", encoding="utf-8") as file_pointer:
        file_pointer.write(
            ordered_yaml_dump(
                obj,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        )


def dump_bench_cfg(bench_cfg, output_dir):
    """
    write bench_cfg as JSON to <output_dir>/bench_cfg.json
    """
    out_path = os.path.join(output_dir, "bench_cfg.json")
    with open(out_path, "w", encoding='utf-8') as fp:
        json.dump(bench_cfg, fp)


def dump_run_cfg(run_cfg, output_dir, exclude_yaml=False):
    """
    write run_cfg as JSON to <output_dir>/run_cfg.json
    """
    out_path = os.path.join(output_dir, "run_cfg.json")

    if exclude_yaml and "yaml" in run_cfg.keys():
        run_cfg = copy.deepcopy(run_cfg)
        del run_cfg["yaml"]

    with open(out_path, "w", encoding='utf-8') as fp:
        json.dump(run_cfg, fp)


def generate_training_dir_name(run_cfg: dict) -> str:
    """Return the name of the training directory."""
    return "bench_" + run_cfg["hash"]


def generate_training_dir_path(run_cfg: dict, bench_cfg: dict) -> str:
    """Generate path for output directory"""
    return os.path.join(
        bench_cfg["output_path"], generate_training_dir_name(run_cfg)
    )


def generate_yaml_cfg_name(run_cfg: dict) -> str:
    """Return the name of the config file."""
    return "config_" + run_cfg["hash"] + ".yaml"


def create_training_dir(run_cfg: dict, bench_cfg: dict) -> list[str, str]:
    """Create training directory."""
    dir_path = generate_training_dir_path(run_cfg, bench_cfg)
    if create_exit_dir(dir_path):
        return [dir_path, ""]

    mf_cfg_name = generate_yaml_cfg_name(run_cfg)
    mf_cfg_path = os.path.join(dir_path, mf_cfg_name)
    for save_path in ["output_dir"]:
        run_cfg["yaml"][save_path] = os.path.join(
            dir_path, run_cfg["yaml"][save_path].strip("./")
        )
    save_yaml(run_cfg["yaml"], mf_cfg_path)
    logger.info(f"Config file is saved here: {mf_cfg_path}")
    return dir_path, mf_cfg_path
