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


import random
import numpy as np
from ruck import *


class OneMaxMinimalModule(EaModule):
    """
    Minimal onemax example
    - The goal is to flip all the bits of a boolean array to True
    - Offspring are generated as bit flipped versions of the previous population
    - Selection tournament is performed between the previous population and the offspring
    """

    # evaluate unevaluated members according to their values
    def evaluate_values(self, values):
        return [v.sum() for v in values]

    # generate 300 random members of size 100 with 50% bits flipped
    def gen_starting_values(self):
        return [np.random.random(100) < 0.5 for _ in range(300)]

    # randomly flip 5% of the bits of each each member in the population
    # the previous population members should never be modified
    def generate_offspring(self, population):
        return [Member(m.value ^ (np.random.random(m.value.shape) < 0.05)) for m in population]

    # selection tournament between population and offspring
    def select_population(self, population, offspring):
        combined = population + offspring
        return [max(random.sample(combined, k=3), key=lambda m: m.fitness) for _ in range(len(population))]


if __name__ == '__main__':
    # create and train the population
    module = OneMaxMinimalModule()
    pop, logbook, halloffame = Trainer(generations=100, progress=True).fit(module)

    print('initial stats:', logbook[0])
    print('final stats:', logbook[-1])
    print('best member:', halloffame.members[0])
