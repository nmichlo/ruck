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

import timeit
from typing import Tuple

import numpy as np

from ruck import Member
from ruck.external.deap import select_nsga2
from ruck.functional import select_nsga2 as select_nsga2_ruck


# ========================================================================= #
# TEST                                                                      #
# ========================================================================= #


# CONFIG:            | JIT:                                                | PYTHON:
# (in=0008 out=0004) | [OLD:  0.000176 NEW: 0.000156s SPEEDUP:  1.126655x] | [OLD:  0.000139 NEW:  0.000198s SPEEDUP: 0.699888x]
# (in=0064 out=0032) | [OLD:  0.002818 NEW: 0.000316s SPEEDUP:  8.913371x] | [OLD:  0.002732 NEW:  0.003151s SPEEDUP: 0.867194x]
# (in=0256 out=0128) | [OLD:  0.040459 NEW: 0.001258s SPEEDUP: 32.161621x] | [OLD:  0.038630 NEW:  0.045156s SPEEDUP: 0.855490x]
# (in=1024 out=0512) | [OLD:  0.672029 NEW: 0.010862s SPEEDUP: 61.872225x] | [OLD:  0.644428 NEW:  0.768074s SPEEDUP: 0.839018x]
# (in=4096 out=2048) | [OLD: 10.511867 NEW: 0.158704s SPEEDUP: 66.235660x] | [OLD: 10.326754 NEW: 12.973584s SPEEDUP: 0.795983x]


if __name__ == '__main__':

    def run_small(
        weights: Tuple[int, ...] = (1, -1),
        seed: int = 42,
    ):
        np.random.seed(seed)

        population = [Member(0, (0.5, 0.5)), Member(1, (0.5, 0.5))] + [Member(i, fitness=tuple(np.random.random(len(weights)))) for i in range(2, 16)]

        orig = np.array([m.fitness for m in population])
        sel0 = np.array([m.fitness for m in select_nsga2(population, 5, weights)])
        sel1 = np.array([m.fitness for m in select_nsga2_ruck(population, 5, weights)])

        print(orig)
        print(sel0)
        print(sel1)


    def run_sweep(
        repeats: int = 3,
        weights: Tuple[int, ...] = (1, -1),
        seed: int = 42,
    ):
        np.random.seed(seed)
        # run sweep
        for N in [8, 2**6, 2**8, 2**10, 2**12]: # , 2**14, 2**16, 2**18, 2**20]:
            # create population
            population = [Member(0, (0.5, 0.5)), Member(1, (0.5, 0.5))]
            population.extend(Member(i, fitness=tuple(np.random.random(len(weights)))) for i in range(2, N))
            # run versions
            time_new = timeit.timeit(lambda: select_nsga2_ruck(population, N//2, weights), number=repeats) / repeats
            time_old = timeit.timeit(lambda:        select_nsga2(population, N//2, weights), number=repeats) / repeats
            print(f'(in={N:04d} out={N//2:04d}) [OLD: {time_old:7f} NEW: {time_new:7f}s SPEEDUP: {time_old/time_new:7f}x]')

    run_small()
    run_sweep()
