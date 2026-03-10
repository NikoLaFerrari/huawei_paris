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
"""run tests against real values"""

import argparse
import os
import sys
import csv
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from tempfile import NamedTemporaryFile
import shutil
import statistics as Stat

from paradise.logger import logger, set_verbose_level
import paradise.parallelize as Par
import paradise.dimensions as Dim
import paradise.common.hardware as Hard
import paradise.debug as Dbg


def make_parall(yaml, device, batch, csv_name):
    """Make the parallelizer"""
    if "telecom" in csv_name:
        return Par.Parallelize(yaml, device, batch, mppb=True, model="Telecom")
    return Par.Parallelize(yaml, device, batch, mppb=True)


def make_paths():
    """Return directories of Paradise"""
    file_path = os.path.dirname(os.path.realpath(__file__))
    return {
        "yaml": os.path.join(file_path, "yamls"),
        "csv": os.path.join(file_path, "results"),
        "output": os.path.join(file_path, "output"),
    }


def anti_perf_regression():
    """Test ensuring ND runs fast"""
    max_duration = 15
    start = time.time()
    paradise = Par.Parallelize(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "yamls",
            "ds",
            "deepseek.yaml",
        ),
        Hard.Machine(1024, 2),
        512,
        Dim.get_dims(["DP", "OP", "MP", "EP", "PP", "MB"]),
    )
    paradise.run_generation_to_ordering(None, top_num=1)
    duration = time.time() - start
    logger.output(
        "Total time taken for performance anti regression is %2.2fs", duration
    )
    if duration > max_duration:
        logger.error("duration should not exceed %ds!", max_duration)
        sys.exit(1)


def treat_total_time(files, paths, temp_peak, global_results):
    """Correlation tests between total profiled time and score"""
    correls = []
    with open(files["csv_peak"], newline="", encoding="utf-8") as csv_file:
        rows = csv.DictReader(csv_file)
        rows = [
            {
                k.strip() if isinstance(k, str) else k: (
                    v.strip() if isinstance(v, str) else v
                )
                for k, v in row.items()
            }
            for row in rows
        ]
        writer = csv.DictWriter(temp_peak, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            treat_row(r, correls, paths, writer)

    average = (
        global_results["peak"]["Average"] / 100,
        Stat.mean(correls),
    )
    median = (
        global_results["peak"]["Median"] / 100,
        Stat.median(correls),
    )

    Dbg.print_diff("Median", median[0], median[1], tabsize=10)
    Dbg.print_diff("Average", average[0], average[1], tabsize=10)

    if median[0] > median[1] or average[0] > average[1]:
        logger.critical("ERROR: New results are worse than before!")
        sys.exit(1)

    logger.info("Results are improved!")
    global_results["peak"]["Average"] = average[1] * 100
    global_results["peak"]["Median"] = median[1] * 100


def treat_row(r, correls, paths, writer):
    """Treat 1 correlation test"""
    logger.debug(
        "test %s, %s, b=%d, d=%d, A%d",
        r["yaml"],
        r["csv"],
        int(r["batch"]),
        int(r["dev_num"]),
        int(r["dev_type"]),
    )
    machine = Hard.Machine(int(r["dev_num"]), int(r["dev_type"]))
    paradise = make_parall(
        os.path.join(paths["yaml"], r["yaml"]),
        machine,
        int(r["batch"]),
        r["csv"],
    )
    correl, top_k, total = paradise.test_from_csv(
        os.path.join(paths["csv"], r["csv"]),
        output_path=paths["output"],
    )

    correls.append(correl)
    Dbg.print_diff(
        r["csv"],
        float(r["correl"]) / 100,
        correl,
        topk=top_k,
        total=total,
    )
    r["correl"] = 100 * correl
    writer.writerow(r)


def treat_components_time(files, paths, temp_json, global_results):
    """Correlation tests between component profiled time and score"""
    comparisons = []
    with open(
        files["csv_class"],
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = csv.DictReader(csv_file)
        rows = [
            {
                k.strip() if isinstance(k, str) else k: (
                    v.strip() if isinstance(v, str) else v
                )
                for k, v in row.items()
            }
            for row in rows
        ]
        for n, r in enumerate(rows):
            logger.debug(
                "test %s, %s, b=%d, d=%d, A%d",
                r["yaml"],
                r["csv"],
                r["batch"],
                r["dev_num"],
                r["dev_type"],
            )
            machine = Hard.Machine(int(r["dev_num"]), int(r["dev_type"]))
            paradise = make_parall(
                os.path.join(paths["yaml"], r["yaml"]),
                machine,
                int(r["batch"]),
                r["csv"],
            )
            correl, distance, top_k, total = (
                paradise.test_from_csv_comm_classified(
                    os.path.join(paths["csv"], r["csv"]),
                    output_path=paths["output"],
                )
            )
            comparisons.append((correl, distance, top_k, total))
            logger.output("File %d is %s", n + 1, r["csv"])
        Dbg.print_correlations_classified(comparisons)

    global_results["last_modification"] = str(
        datetime.now(ZoneInfo(global_results["zone"]))
    )
    json.dump(global_results, temp_json, indent=4)


def anti_estimation_regression(change):
    """Tests ensuring ND estimation accuracy does not decrease"""
    paths = make_paths()

    files = {
        "csv_peak": os.path.join(paths["csv"], "results_peak.csv"),
        "csv_class": os.path.join(paths["csv"], "results_classified.csv"),
        "json_global": os.path.join(paths["csv"], "global_results.json"),
    }

    with NamedTemporaryFile(
        "w",
        newline="",
        delete=False,
        encoding="utf-8",
    ) as temp_json:
        with NamedTemporaryFile(
            "w",
            newline="",
            delete=False,
            encoding="utf-8",
        ) as temp_peak:
            with open(
                files["json_global"],
                "r",
                encoding="utf-8",
            ) as gr:
                global_results = json.load(gr)
                treat_total_time(files, paths, temp_peak, global_results)
                treat_components_time(files, paths, temp_json, global_results)

    if change:
        logger.output("Saving new results!")
        shutil.move(temp_peak.name, files["csv_peak"])
        shutil.move(temp_json.name, files["json_global"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="PARallelize ALL dimensions",
        description=("Provides a number of devices per parallel dimensions"),
        epilog="",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=2,
        help="Level of verbosity in range [0,6], "
        "0 being no output and 5 being debug level output",
    )
    parser.add_argument(
        "-c",
        "--change_if_better",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Will update result files if better (supposed to be kept for CI)",
    )
    # parser.add_argument(
    #     "-i",
    #     "--plot_idle",
    #     type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
    #     default=True,
    #     help="Whether to plot idle time for detailed run",
    # )

    args = parser.parse_args()
    set_verbose_level(args.verbosity)
    # plot_idle = args.plot_idle

    anti_estimation_regression(args.change_if_better)
    anti_perf_regression()
