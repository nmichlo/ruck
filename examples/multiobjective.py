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
from matplotlib import pyplot as plt

from ruck import *
from ruck.functional import select_nsga2


class MultiObjectiveMinimalModule(EaModule):
    """
    Minimal onemax example
    - The goal is to flip all the bits of a boolean array to True
    - Offspring are generated as bit flipped versions of the previous population
    - Selection tournament is performed between the previous population and the offspring
    """

    # evaluate unevaluated members
    def evaluate_values(self, values):
        return [(y - x**2, x - y**2) for (x, y) in values]

    # generate values in the range [-1, 1]
    def gen_starting_values(self):
        return [np.random.random(2) * 2 - 1 for _ in range(100)]

    # randomly offset the members by a small amount
    def generate_offspring(self, population):
        return [Member(np.clip(m.value + np.random.randn(2) * 0.05, -1, 1)) for m in population]

    # apply nsga2 to population, which tries to maintain a diverse set of solutions
    def select_population(self, population, offspring):
        return select_nsga2(population + offspring, len(population))


if __name__ == '__main__':
    # create and train the population
    module = MultiObjectiveMinimalModule()
    pop, logbook, halloffame = Trainer(generations=100, progress=True).fit(module)

    print('initial stats:', logbook[0])
    print('final stats:', logbook[-1])
    print('best member:', halloffame.members[0])

    # plot path
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
    # plot points
    ax0.set_title('Pareto Optimal Values')
    ax0.scatter(*zip(*(m.value for m in pop)))
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    # plot pareto optimal solution
    ax1.set_title('Pareto Optimal Scores')
    ax1.scatter(*zip(*(m.fitness for m in pop)))
    ax1.set_xlabel('Distances')
    ax1.set_ylabel('Smoothness')
    # display
    fig.tight_layout()
    plt.show()
