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
import csv
import sys

from paradise.logger import logger, set_verbose_level
import paradise.common.hardware as Hard
import paradise.debug as Dbg
import paradise.run_test as Test


def treat_row(r, paths):
    """Validate 1 file"""
    logger.debug(
        "test %s, %s, b=%d, d=%d, A%d",
        r["yaml"],
        r["csv"],
        int(r["batch"]),
        int(r["dev_num"]),
        int(r["dev_type"]),
    )
    machine = Hard.Machine(int(r["dev_num"]), int(r["dev_type"]))
    paradise = Test.make_parall(
        os.path.join(paths["yaml"], r["yaml"]),
        machine,
        int(r["batch"]),
        r["csv"],
    )
    csv_f = os.path.join(paths["csv"], r["csv"])
    _, top_k, total = paradise.test_from_csv(
        csv_f,
        output_path=paths["output"],
    )

    dims = Dbg.get_diff_dims(csv_f)
    best_pos = -min(top_k, 0) + 1
    logger.output(
        "%s\t%s\t%d\t%d\t%d\t%d\t%s",
        paradise.model_name,
        str(paradise.machine.device),
        paradise.machine.number,
        paradise.global_batch_size,
        total,
        best_pos,
        str(dims),
    )
    if best_pos != 1:
        logger.error("Best configuration not found for test %s", {r["csv"]})
        sys.exit(1)


def validate():
    """Validate all files"""
    paths = Test.make_paths()
    logger.output(
        "Model name\tMachine\tDevices\tGBS\tTest\tTopPos\tVarying Dimensions"
    )
    with open(
        os.path.join(paths["csv"], "results_validation.csv"),
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
        for row in rows:
            treat_row(row, paths)


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

    args = parser.parse_args()
    set_verbose_level(args.verbosity)
    validate()
