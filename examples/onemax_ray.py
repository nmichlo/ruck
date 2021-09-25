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

"""
OneMax parallel example using ray's object store.

8 bytes * 1_000_000 * 128 members ~= 128 MB of memory to store this population.
This is quite a bit of processing that needs to happen! But using ray
and its object store we can do this efficiently!
"""

from functools import partial
import numpy as np
import ray
from ruck import *
from ruck.util import *


class OneMaxRayModule(EaModule):

    def __init__(
        self,
        population_size: int = 300,
        member_size: int = 100,
        p_mate: float = 0.5,
        p_mutate: float = 0.5,
    ):
        self.save_hyperparameters()
        # implement the required functions for `EaModule`
        # - decorate the functions with `ray_refs_wrapper` which
        #   automatically `ray.get` arguments and `ray.put` returned results
        self.generate_offspring, self.select_population = R.factory_simple_ea(
            mate_fn=ray_refs_wrapper(R.mate_crossover_1d, iter_results=True),
            mutate_fn=ray_refs_wrapper(partial(R.mutate_flip_bit_groups, p=0.05)),
            select_fn=partial(R.select_tournament, k=3),  # OK to compute locally, because we only look at the fitness
            p_mate=self.hparams.p_mate,
            p_mutate=self.hparams.p_mutate,
            map_fn=ray_map,  # specify the map function to enable multiprocessing
        )

    def evaluate_values(self, values):
        # values is a list of `ray.ObjectRef`s not `np.ndarray`s
        # ray_map automatically converts np.sum to a `ray.remote` function which
        # automatically handles `ray.get`ing of `ray.ObjectRef`s passed as arguments
        return ray_map(np.sum, values)

    def gen_starting_values(self):
        # generate objects and place in ray's object store
        return [
            ray.put(np.random.random(self.hparams.member_size) < 0.5)
            for i in range(self.hparams.population_size)
        ]


if __name__ == '__main__':
    # initialize ray to use the specified system resources
    ray.init()

    # create and train the population
    module = OneMaxRayModule(population_size=128, member_size=1_000_000)
    pop, logbook, halloffame = Trainer(generations=100, progress=True).fit(module)

    print('initial stats:', logbook[0])
    print('final stats:', logbook[-1])
    print('best member:', halloffame.members[0])
