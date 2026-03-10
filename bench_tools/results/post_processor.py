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

"""handling and extending benchmark result processing"""

import copy
from abc import ABC, abstractmethod

from bench_tools.results.bench_result import BenchResult


class PostProcessor(ABC):
    """ base class for benchmark post-processing passes """
    registry = {}
    default_config = {}

    def __init__(self, post_processor_config: dict = None):
        super().__init__()
        self.config = copy.deepcopy(self.default_config)
        if post_processor_config is not None:
            self.config.update(post_processor_config)

    @classmethod
    def register(cls, subclass):
        """
        class decorator to register a PostProcessor subclass by
        'name' attribute
        """
        name = subclass.name
        if name in cls.registry:
            raise ValueError(f"Duplicate class name: {name}")
        cls.registry[name] = subclass
        return subclass

    @classmethod
    def get_instance_by_name(
        cls, name: str, post_processor_config: dict
    ) -> "PostProcessor":
        """
        instantiate a registered post-processor by name,
        returns None if name not found
        """
        if name in cls.registry:
            return cls.registry[name](post_processor_config)
        return None

    @abstractmethod
    def execute(self, bench_result: BenchResult) -> None:
        """ not implemented """
        pass

    def execute_multiple(self, bench_results: list[BenchResult]) -> None:
        """ not implemented """
        pass

    def display(self, bench_result: BenchResult) -> None:
        """ not implemented """
        pass


# Example for a post-processor subclass
#
# @PostProcessor.register
# class examplePostProcessor(PostProcessor):
#    name = "example"
#    default_config = {}
#
#    def execute(self, bench_result: BenchResult) -> None:
#        pass
#
#    # optional
#    def display(self, bench_result: BenchResult) -> None:
#        pass
#
# Remember to add the import in run_utils.py run_post_processors() to register the class:
# from .example.py import examplePostProcessor
