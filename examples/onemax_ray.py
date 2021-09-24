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

import logging
from functools import wraps
from typing import Any
from typing import List

import numpy as np
import ray

from ruck import *
from ruck import EaModule
from ruck import Population
from ruck.util import chained
from ruck.util import ray_map
from ruck.util import Timer
from ruck.util._iter import ipairs

from ruck.util._iter import itake_random
from ruck.util._iter import random_map
from ruck.util._iter import random_map_pairs
from ruck.util._iter import replaced
from ruck.util._iter import transposed


# ========================================================================= #
# Module                                                                    #
# ========================================================================= #


def ray_store(get: bool = True, put: bool = True, iter_results: bool = False):
    def wrapper(fn):
        @wraps(fn)
        def inner(*args):
            # get values from object store
            if get:
                args = (ray.get(v) for v in args)
            # call function
            result = fn(*args)
            # store values in the object store
            if put:
                if iter_results:
                    result = tuple(ray.put(v) for v in result)
                else:
                    result = ray.put(result)
            # done!
            return result
        return inner
    return wrapper


def member_values(unwrap: bool = True, wrap: bool = True, iter_results: bool = False):
    def wrapper(fn):
        @wraps(fn)
        def inner(*args):
            # unwrap member values
            if unwrap:
                args = (m.value for m in args)
            # call function
            result = fn(*args)
            # wrap values withing members again
            if wrap:
                if iter_results:
                    result = tuple(Member(v) for v in result)
                else:
                    result = Member(result)
            # done!
            return result
        return inner
    return wrapper


@member_values(iter_results=True)
@ray_store(iter_results=True)
def mate(a, b):
    return R.mate_crossover_1d(a, b)


@member_values()
@ray_store()
def mutate(v):
    return R.mutate_flip_bit_types(v, p=0.05)


class OneMaxModule(EaModule):

    def __init__(
        self,
        population_size: int = 300,
        member_size: int = 100,
        p_mate: float = 0.5,
        p_mutate: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

    def gen_starting_values(self):
        for _ in range(self.hparams.population_size):
            yield ray.put(np.random.random(self.hparams.member_size) < 0.5)

    def generate_offspring(self, population: Population) -> Population:
        # Same as deap.algorithms.eaSimple which uses deap.algorithms.varAnd
        offspring = list(population)
        np.random.shuffle(offspring)
        offspring = random_map_pairs(mate, offspring, p=self.hparams.p_mate, map_fn=ray_map)
        offspring = random_map(mutate, offspring, p=self.hparams.p_mutate,   map_fn=ray_map)
        # Done!
        return offspring

    def select_population(self, population: Population, offspring: Population) -> Population:
        return R.select_tournament(population + offspring, len(population), k=3)  # TODO: tools.selNSGA2

    def evaluate_values(self, values: List[Any]) -> List[float]:
        return ray_map(np.mean, values)


# ========================================================================= #
# Main                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    # about 18x faster than deap's numpy onemax example (0.145s vs 2.6s)
    # -- https://github.com/DEAP/deap/blob/master/examples/ga/onemax_numpy.py

    logging.basicConfig(level=logging.INFO)

    ray.init(num_cpus=64)

    with Timer('ruck:trainer'):
        module = OneMaxModule(population_size=512, member_size=1_000_000)
        pop, logbook, halloffame = Trainer(generations=1000, progress=True).fit(module)

    print('initial stats:', logbook[0])
    print('final stats:', logbook[-1])


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
