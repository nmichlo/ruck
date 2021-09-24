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
from typing import Any
from typing import List

import numpy as np

from ruck import *
from ruck import EaModule
from ruck import Population
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

    def evaluate_values(self, values: List[Any]):
        # this is a large reason why the deap version is slow,
        # it does not make use of numpy operations
        return map(np.sum, values)

    def gen_starting_values(self) -> Population:
        for _ in range(self.hparams.population_size):
            yield np.random.random(self.hparams.member_size) < 0.5

    def generate_offspring(self, population: Population) -> Population:
        # Same as deap.algorithms.eaSimple which uses deap.algorithms.varAnd
        offspring = R.select_tournament(population, len(population), k=3)  # tools.selNSGA2
        # vary population
        return R.apply_mate_and_mutate(
            population=offspring,
            mate_fn=R.mate_crossover_1d,
            mutate_fn=lambda a: R.mutate_flip_bits(a, p=0.05),
            p_mate=self.hparams.p_mate,
            p_mutate=self.hparams.p_mutate,
        )

    def select_population(self, population: Population, offspring: Population) -> Population:
        # Same as deap.algorithms.eaSimple
        return offspring


# ========================================================================= #
# Main                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    # about 18x faster than deap's numpy onemax example (0.145s vs 2.6s)
    # -- https://github.com/DEAP/deap/blob/master/examples/ga/onemax_numpy.py

    logging.basicConfig(level=logging.INFO)

    with Timer('ruck:trainer'):
        module = OneMaxModule(population_size=300, member_size=100)
        pop, logbook, halloffame = Trainer(generations=40, progress=False).fit(module)

    print('initial stats:', logbook[0])
    print('final stats:', logbook[-1])


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
