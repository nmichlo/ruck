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


# ========================================================================= #
# TEST                                                                      #
# ========================================================================= #


import timeit
from typing import Tuple

import numpy as np

from ruck import Member
from ruck.external.deap import select_nsga2
from ruck.external.deap import select_nsga2_custom
from ruck.util import Timer


if __name__ == '__main__':

    def run_small(
        weights: Tuple[int, ...] = (1, -1),
        seed: int = 42,
    ):
        np.random.seed(seed)

        population = [Member(0, (0.5, 0.5)), Member(1, (0.5, 0.5))] + [Member(i, fitness=tuple(np.random.random(len(weights)))) for i in range(2, 16)]

        orig = np.array([m.fitness for m in population])
        sel0 = np.array([m.fitness for m in select_nsga2(population, 5, weights)])
        sel1 = np.array([m.fitness for m in select_nsga2_custom(population, 5, weights)])

        print(orig)
        print(sel0)
        print(sel1)


    def run_sweep(
        repeats: int = 1,
        mul_iters: int = 1,
        weights: Tuple[int, ...] = (1, -1),
        seed: int = 42,
    ):
        np.random.seed(seed)
        # run sweep
        for N in [8, 2**6, 2**8, 2**10, 2**12]: # , 2**14, 2**16, 2**18, 2**20]:
            # create population
            with Timer(f'population-{N}') as t:
                population = [Member(0, (0.5, 0.5)), Member(1, (0.5, 0.5))]
                population.extend(Member(i, fitness=tuple(np.random.random(len(weights)))) for i in range(2, N))
            # run versions
            time_new = timeit.timeit(lambda: select_nsga2_custom(population, N//2, weights), number=repeats) / repeats * mul_iters
            print(f'- NEW: {N}-{N//2}: {time_new}sec for {mul_iters} iters')
            time_old = timeit.timeit(lambda:        select_nsga2(population, N//2, weights), number=repeats) / repeats * mul_iters
            print(f'- OLD: {N}-{N//2}: {time_old}sec for {mul_iters} iters')
            print(f'* speedup: {time_old/time_new:3f}x')

            print()

    run_small()
    run_sweep()
