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
from ruck.external.deap import select_nsga2
from ruck.external.deap import select_nsga2_custom


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
        Member(None, fitness=tuple(np.random.random(fitness_size)))
        for _ in range(population_size)
    ]
    # select
    sel_ref = select_nsga2(population, sel_num, weights)
    sel_lib = select_nsga2_custom(population, sel_num, weights)
    # checks
    assert sel_ref == sel_lib

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
