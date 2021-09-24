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
import random
from typing import Any
from typing import List
from typing import Tuple

import numpy as np
import ray
from ray import ObjectRef

from ruck import *
from ruck import EaModule
from ruck import Population
from ruck.util import chained
from ruck.util import ray_map
from ruck.util import Timer


# ========================================================================= #
# Module                                                                    #
# ========================================================================= #


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

    def gen_starting_population(self) -> Population:
        # 2.0317113399505615
        return [
            Member(ray.put(np.random.random(self.hparams.member_size) < 0.5))
            for _ in range(self.hparams.population_size)
        ]

    def generate_offspring(self, population: Population) -> Population:
        # HACK
        # 0.027140140533447266
        # population = [Member(ray.get(m.value), m.fitness) for m in population]
        # Same as deap.algorithms.eaSimple which uses deap.algorithms.varAnd
        # 0.0007593631744384766
        offspring = R.select_tournament(population, len(population), k=3)  # tools.selNSGA2
        # vary population
        # 0.7187347412109375

        @ray.remote
        def mate_crossover_1d(a, b) -> Tuple[ObjectRef, ObjectRef]:
            a, b = R.mate_crossover_1d(a, b)
            return ray.put(a), ray.put(b)

        @ray.remote
        def mutate_flip_bits(a) -> ObjectRef:
            a = R.mutate_flip_bits(a, p=0.05)
            return ray.put(a)

        with Timer('vary'):
            # mate
            random.shuffle(offspring)
            futures, positions = [], []
            for i, (a, b) in enumerate(zip(offspring[0::2], offspring[1::2])):
                if random.random() < self.hparams.p_mate:
                    futures.append(mate_crossover_1d.remote(a.value, b.value))
                    positions.append(i)
            for i, (a, b) in zip(positions, ray.get(futures)):
                offspring[i*2+0] = Member(a)  # why does this step slow things down so much?
                offspring[i*2+1] = Member(b)  # why does this step slow things down so much?

            # mutate
            futures, positions = [], []
            for i, a in enumerate(offspring):
                if random.random() < self.hparams.p_mutate:
                    futures.append(mutate_flip_bits.remote(a.value))
            for i, a in zip(positions, ray.get(futures)):
                print(a)
                offspring[i] = Member(a)  # why does this step slow things down so much?

        # offspring = R.apply_mate_and_mutate(
        #     population=offspring,
        #     mate_fn=lambda a, b: R.mate_crossover_1d,
        #     mutate_fn=lambda a: ray.put(R.mutate_flip_bits(ray.get(a), p=0.05)),
        #     p_mate=self.hparams.p_mate,
        #     p_mutate=self.hparams.p_mutate,
        # )
        # HACK
        # 0.13915061950683594
        # offspring = [Member(ray.put(m.value), m.fitness_unsafe) for m in offspring]
        # done
        return offspring

    def select_population(self, population: Population, offspring: Population) -> Population:
        # Same as deap.algorithms.eaSimple
        return offspring

    def evaluate_values(self, values: List[Any]) -> List[float]:
        # 0.1165781021118164
        return ray_map(np.mean, values)


# @ray.remote
# def evaluate(value):
#     return value.std()
    # return [ray.get(value_id).std() for value_id in values]

# @ray.remote
# class RayWorker(object):
#
#     def gen_starting_population(self) -> Population:
#         pass
#
#     def generate_offspring(self, population: Population) -> Population:
#         pass
#
#     def select_population(self, population: Population, offspring: Population) -> Population:
#         pass
#
#     def evaluate_values(self, values: List[Any]) -> List[float]:
#         pass


# class RayManager():
#
#     def __init__(self, num_workers: int = None):
#         if num_workers is None:
#             num_workers = ray.available_resources().get('CPU', 1)


# ========================================================================= #
# Main                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    # about 18x faster than deap's numpy onemax example (0.145s vs 2.6s)
    # -- https://github.com/DEAP/deap/blob/master/examples/ga/onemax_numpy.py

    logging.basicConfig(level=logging.INFO)

    ray.init(num_cpus=128)

    with Timer('ruck:trainer'):
        module = OneMaxModule(population_size=512, member_size=1_000_000)
        pop, logbook, halloffame = Trainer(generations=1000, progress=True).fit(module)

    print('initial stats:', logbook[0])
    print('final stats:', logbook[-1])


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
