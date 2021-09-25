#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
import re
from typing import Generic
from typing import List
from typing import Optional
from typing import TypeVar

import numpy as np


# ========================================================================= #
# Members                                                                   #
# ========================================================================= #


class MemberIsNotEvaluatedError(Exception):
    pass


class MemberAlreadyEvaluatedError(Exception):
    pass


T = TypeVar('T')


_RE_WHITESPACE = re.compile(r'\s\s+')


class Member(Generic[T]):

    def __init__(self, value: T, fitness: float = None):
        self._value = value
        self._fitness = None
        # set fitness
        if fitness is not None:
            self.fitness = fitness

    @property
    def value(self) -> T:
        return self._value

    @property
    def fitness_unsafe(self) -> Optional[float]:
        return self._fitness

    @property
    def fitness(self) -> float:
        if not self.is_evaluated:
            raise MemberIsNotEvaluatedError('The member has not been evaluated, the fitness has not yet been set.')
        return self._fitness

    @fitness.setter
    def fitness(self, fitness: float):
        if self.is_evaluated:
            raise MemberAlreadyEvaluatedError('The member has already been evaluated, the fitness can only ever be set once. Create a new member instead!')
        if np.isnan(fitness):
            raise ValueError('fitness values cannot be NaN, this is an error!')
        self._fitness = float(fitness)

    def set_fitness(self, fitness: float) -> 'Member[T]':
        self.fitness = fitness
        return self

    @property
    def is_evaluated(self) -> bool:
        return (self._fitness is not None)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        value_str = _RE_WHITESPACE.sub(' ', repr(self.value))
        # cut short
        if len(value_str) > 33:
            value_str = f'{value_str[:14].rstrip(" ")} ... {value_str[-14:].lstrip(" ")}'
        # get fitness
        fitness_str = f', {self.fitness}' if self.is_evaluated else ''
        # combine
        return f'{self.__class__.__name__}({value_str}{fitness_str})'


# ========================================================================= #
# Population                                                                #
# ========================================================================= #


Population = List[Member[T]]


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
