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


import functools
import logging
from typing import List

import numpy as np
import psutil
import ray
from ray import ObjectRef

from ruck import EaModule
from ruck import Population
from ruck import R
from ruck import Trainer
from ruck.util import ray_mapped
from ruck.util import Timer
from ruck.util._ray import ray_refs_handler


# ========================================================================= #
# Module                                                                    #
# ========================================================================= #


class OneMaxRayModule(EaModule[ObjectRef]):

    # trick pycharm overrides error checking against `EaModule`
    # it doesn't like that we set the values in the constructor!
    generate_offspring = None
    select_population = None

    def __init__(
        self,
        population_size: int = 300,
        member_size: int = 100,
        p_mate: float = 0.5,
        p_mutate: float = 0.5,
    ):
        # save the arguments to the .hparams property. values are taken from the
        # local scope so modifications can be captured if the call to this is delayed.
        self.save_hyperparameters()
        # implement the required functions for `EaModule`
        self.generate_offspring, self.select_population = R.factory_simple_ea(
            mate_fn=ray_refs_handler(R.mate_crossover_1d, iter_results=True),
            mutate_fn=ray_refs_handler(functools.partial(R.mutate_flip_bits, p=0.05)),
            select_fn=functools.partial(R.select_tournament, k=3),  # tools.selNSGA2
            p_mate=self.hparams.p_mate,
            p_mutate=self.hparams.p_mutate,
            map_fn=ray_mapped,
        )

    def evaluate_values(self, values: List[ObjectRef]) -> List[float]:
        # this is a large reason why the deap version is slow,
        # it does not make use of numpy operations
        return ray_mapped(np.sum, values)

    def gen_starting_values(self) -> Population[ObjectRef]:
        return [
            ray.put(np.random.random(self.hparams.member_size) < 0.5)
            for i in range(self.hparams.population_size)
        ]


# ========================================================================= #
# Main                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    # about 15x faster than deap's numpy onemax example (0.17s vs 2.6s)
    # -- https://github.com/DEAP/deap/blob/master/examples/ga/onemax_numpy.py

    logging.basicConfig(level=logging.INFO)

    ray.init(num_cpus=min(psutil.cpu_count(), 16))

    with Timer('ruck:trainer'):
        module = OneMaxRayModule(population_size=512, member_size=1_000_000)
        pop, logbook, halloffame = Trainer(generations=100, progress=False).fit(module)

    print('initial stats:', logbook[0])
    print('final stats:', logbook[-1])


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
