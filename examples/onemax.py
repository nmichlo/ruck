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
OneMax serial example based on:
https://github.com/DEAP/deap/blob/master/examples/ga/onemax_numpy.py
"""

import functools
import numpy as np
from ruck import *


class OneMaxModule(EaModule):

    def __init__(
        self,
        population_size: int = 300,
        offspring_num: int = None,  # offspring_num (lambda) is automatically set to population_size (mu) when `None`
        member_size: int = 100,
        p_mate: float = 0.5,
        p_mutate: float = 0.5,
        ea_mode: str = 'simple'
    ):
        # save the arguments to the .hparams property. values are taken from the
        # local scope so modifications can be captured if the call to this is delayed.
        self.save_hyperparameters()
        # implement the required functions for `EaModule`
        self.generate_offspring, self.select_population = R.make_ea(
            mode=self.hparams.ea_mode,
            offspring_num=self.hparams.offspring_num,
            mate_fn=R.mate_crossover_1d,
            mutate_fn=functools.partial(R.mutate_flip_bit_groups, p=0.05),
            select_fn=functools.partial(R.select_tournament, k=3),
            p_mate=self.hparams.p_mate,
            p_mutate=self.hparams.p_mutate,
        )

    def evaluate_values(self, values):
        return map(np.sum, values)

    def gen_starting_values(self) -> Population:
        return [
            np.random.random(self.hparams.member_size) < 0.5
            for i in range(self.hparams.population_size)
        ]


if __name__ == '__main__':
    # create and train the population
    module = OneMaxModule(population_size=300, member_size=100)
    pop, logbook, halloffame = Trainer(generations=40, progress=True).fit(module)

    print('initial stats:', logbook[0])
    print('final stats:', logbook[-1])
    print('best member:', halloffame.members[0])
