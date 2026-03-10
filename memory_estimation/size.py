# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Memory arithmetic"""

from enum import Enum, auto

from memory_estimation.logger import logger


class Unit(Enum):
    """Memory units"""

    B = auto()
    KB = auto()
    MB = auto()
    GB = auto()

    @classmethod
    def from_string(cls, string):
        """Constructor from string"""
        if string.upper() == "GB":
            return cls.GB
        if string.upper() == "MB":
            return cls.MB
        if string.upper() == "KB":
            return cls.KB
        if string.upper() == "B":
            return cls.B
        logger.warning("Memory unit was not specified. Byte is taken")
        return cls.B

    def __str__(self):
        if self == self.GB:
            return "GB"
        if self == self.MB:
            return "MB"
        if self == self.KB:
            return "KB"
        return "B"

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Memory:
    """Memory units"""

    size: float
    unit: Unit

    def __init__(self, size, unit):
        self.size = size
        self.unit = unit

    @classmethod
    def from_string(cls, string):
        """Constructor from string"""
        numeric = "0123456789-."
        i = 0
        for i, c in enumerate(string):
            if c not in numeric:
                break
        memory = cls(float(string[:i]), Unit.from_string(string[i:].lstrip()))
        if i <= 0 or i >= len(string):
            logger.critical("Memory seems to be wrong: %s", str(memory))
        return memory

    @classmethod
    def from_b(cls, size: float):
        """Constructor from bytes"""
        return cls(size, Unit.B)

    @classmethod
    def from_kb(cls, size: float):
        """Constructor from kilo bytes"""
        return cls(size, Unit.KB)

    @classmethod
    def from_mb(cls, size: float):
        """Constructor from mega bytes"""
        return cls(size, Unit.MB)

    @classmethod
    def from_gb(cls, size: float):
        """Constructor from giga bytes"""
        return cls(size, Unit.GB)

    @classmethod
    def zero(cls):
        """Constructor for memory size 0"""
        return cls(0, Unit.B)

    def __str__(self):
        if self.unit == Unit.GB:
            return f"{self.size:.2f}{self.unit}"
        return f"{self.size:.0f}{self.unit}"

    def to(self, unit: Unit):
        """Convert to the given unit"""
        diff = unit.value - self.unit.value
        self.size /= pow(1024, diff)
        self.unit = unit
        return self

    def to_gb(self):
        """Convert to GB"""
        return self.to(Unit.GB)

    def to_mb(self):
        """Convert to MB"""
        return self.to(Unit.MB)

    def to_kb(self):
        """Convert to KB"""
        return self.to(Unit.KB)

    def to_b(self):
        """Convert to Byte"""
        return self.to(Unit.B)

    def increase(self, mem):
        """Addition of 2 memory sizes"""
        if self.__class__ is not mem.__class__:
            return NotImplemented
        mem.to(self.unit)
        self.size += mem.size
        return self

    def __add__(self, mem):
        """Addition of 2 memory sizes"""
        if self.__class__ is not mem.__class__:
            return NotImplemented
        new_mem = Memory(self.size, self.unit)
        new_mem.to(mem.unit)
        new_mem.size += mem.size
        if new_mem.unit < self.unit:
            return new_mem
        return new_mem.to(self.unit)

    def decrease(self, mem):
        """Substraction of 2 memory sizes"""
        if self.__class__ is not mem.__class__:
            return NotImplemented
        mem.to(self.unit)
        self.size -= mem.size
        return self

    def __sub__(self, mem):
        """Substraction of 2 memory sizes"""
        if self.__class__ is not mem.__class__:
            return NotImplemented
        new_mem = Memory(self.size, self.unit)
        new_mem.to(mem.unit)
        new_mem.size -= mem.size
        if new_mem.unit < self.unit:
            return new_mem
        return new_mem.to(self.unit)

    def __lt__(self, mem):
        """Comparison of 2 memory sizes"""
        if self.__class__ is not mem.__class__:
            return NotImplemented
        mem.to(self.unit)
        return self.size < mem.size

    def __le__(self, mem):
        """Comparison of 2 memory sizes"""
        if self.__class__ is not mem.__class__:
            return NotImplemented
        mem.to(self.unit)
        return self.size <= mem.size

    def __abs__(self):
        """Absolute value"""
        if self.size < 0:
            self.size = -self.size
        return self
