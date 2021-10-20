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


import numpy as np
import pytest

from ruck import Member
from ruck.functional import select_nsga2 as select_nsga2_ruck
from ruck.external.deap import select_nsga2 as select_nsga2_deap


# ========================================================================= #
# TEST                                                                      #
# ========================================================================= #


@pytest.mark.parametrize(['population_size', 'sel_num', 'fitness_size', 'weights'], [
    # basic
    (0, 0, 2, (1, 1)),
    (1, 0, 2, (1, 1)),
    # (0, 1, 2, (1, 1)),
    (1, 1, 2, (1, 1)),
    # larger
    (10, 0,  2, (1, 1)),
    (10, 1,  2, (1, 1)),
    (10, 5,  2, (1, 1)),
    (10, 9,  2, (1, 1)),
    (10, 10, 2, (1, 1)),
    # (10, 11, 2, (1, 1)),
    # (10, 20, 2, (1, 1)),
    # weights
    (10, 5, 2, ( 1,  1)),
    (10, 5, 2, (-1,  1)),
    (10, 5, 2, ( 1, -1)),
    (10, 5, 2, (-1, -1)),
    (10, 5, 3, (1, -1, 1)),
    (10, 5, 4, (1, -1, 1, -1)),
    (10, 5, 1, (1,)),
    (10, 5, 1, (-1,)),
])
def test(population_size, sel_num, fitness_size, weights):
    np.random.seed(42)
    # generate population
    population = [
        Member(i, fitness=tuple(np.random.randint(5, size=fitness_size)))
        for i in range(population_size)
    ]
    # select
    sel_deap = select_nsga2_deap(population, sel_num, weights)
    sel_ruck = select_nsga2_ruck(population, sel_num, weights)
    # checks
    assert [m.value for m in sel_deap] == [m.value for m in sel_ruck]
    assert [m.fitness for m in sel_deap] == [m.fitness for m in sel_ruck]
    assert sel_deap == sel_ruck


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
