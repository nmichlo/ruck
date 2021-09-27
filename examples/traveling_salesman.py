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

import random
import numpy as np
import matplotlib.pyplot as plt
from ruck import *


class TravelingSalesmanModule(EaModule):

    def __init__(self, points, num_individuals: int = 128, closed_path=False):
        self.num_individuals = int(num_individuals)
        self.points = np.array(points)
        self.num_points = len(self.points)
        self.closed_path = bool(closed_path)
        # checks
        assert self.points.ndim == 2
        assert self.num_points > 0
        assert self.num_individuals > 0

    # OVERRIDE

    def gen_starting_values(self):
        values = [np.arange(self.num_points) for _ in range(self.num_individuals)]
        [np.random.shuffle(v) for v in values]
        return values

    def generate_offspring(self, population):
        # there are definitely much better ways to do this
        return [Member(self._two_opt_swap(random.choice(population).value)) for _ in range(self.num_individuals)]

    def evaluate_values(self, values):
        # we negate because we want to  minimize dist
        return [-self._get_dist(v) for v in values]

    def select_population(self, population, offspring):
        return R.select_tournament(population + offspring, len(population), k=3)

    # HELPER

    def _two_opt_swap(self, idxs):
        i, j = np.random.randint(0, self.num_points, 2)
        i, j = min(i, j), max(i, j)
        nidxs = np.concatenate([idxs[:i], idxs[i:j][::-1], idxs[j:]])
        return nidxs

    def _get_dist(self, value):
        if self.closed_path:
            idxs_from, idxs_to = value, np.roll(value, -1)
        else:
            idxs_from, idxs_to = value[:-1], value[1:]
        # compute dist
        return np.sum(np.linalg.norm(self.points[idxs_from] - self.points[idxs_to], ord=2, axis=-1))

    def get_plot_points(self, value):
        idxs = value.value if isinstance(value, Member) else value
        # handle case
        if self.closed_path:
            idxs = np.concatenate([idxs, [idxs[0]]])
        # get consecutive points
        xs, ys = self.points[idxs].T
        return xs, ys


if __name__ == '__main__':
    # determinism
    random.seed(42)
    np.random.seed(42)
    # get points
    points = np.random.rand(72, 2)
    # train
    module = TravelingSalesmanModule(points=points, num_individuals=128, closed_path=False)
    population, logbook, halloffame = Trainer(generations=1024).fit(module)

    # plot path
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(*module.get_plot_points(halloffame[0]))
    plt.show()
