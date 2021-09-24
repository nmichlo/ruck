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

import random
from functools import wraps
from typing import Callable

from ruck._member import Population


# ========================================================================= #
# Select Helper                                                             #
# ========================================================================= #


SelectFnHint = Callable[[Population, int], Population]


def check_selection(fn):
    @wraps(fn)
    def wrapper(population: Population, num: int):
        selected = fn(population, num)
        assert selected is not population, f'Select function: {fn} should return a new list'
        assert len(selected) == num, f'Select function: {fn} returned an incorrect number of elements, got: {len(selected)}, should be: {num}'
        return selected
    return wrapper


# ========================================================================= #
# Select                                                                    #
# ========================================================================= #


@check_selection
def select_best(population: Population, num: int) -> Population:
    return sorted(population, key=lambda m: m.fitness, reverse=True)[:num]


@check_selection
def select_worst(population: Population, num: int) -> Population:
    return sorted(population, key=lambda m: m.fitness, reverse=False)[:num]


@check_selection
def select_random(population: Population, num: int) -> Population:
    return random.sample(population, k=num)


@check_selection
def select_tournament(population: Population, num: int, k: int = 3) -> Population:
    key = lambda m: m.fitness
    return [
        max(random.sample(population, k=k), key=key)
        for _ in range(num)
    ]


# ========================================================================= #
# Selection                                                                 #
# ========================================================================= #
