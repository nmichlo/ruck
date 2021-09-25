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

import itertools
import logging
from typing import Generic
from typing import Iterator
from typing import Tuple
from typing import TypeVar

import numpy as np
from tqdm import tqdm

from ruck._history import HallOfFame
from ruck._history import Logbook
from ruck._history import StatsGroup
from ruck._member import Member
from ruck._member import Population
from ruck._module import EaModule


log = logging.getLogger(__name__)


T = TypeVar('T')


# ========================================================================= #
# Utils Trainer                                                             #
# ========================================================================= #


def _check_population(population: Population[T], required_size: int) -> Population[T]:
    assert len(population) > 0, 'population must not be empty'
    assert len(population) == required_size, 'population size is invalid'
    assert all(isinstance(member, Member) for member in population), 'items in population are not members'
    return population


# ========================================================================= #
# Evaluate Invalid                                                          #
# ========================================================================= #


def _evaluate_unevaluated(module: EaModule[T], members: Population[T]) -> int:
    # get unevaluated members
    unevaluated = [m for m in members if not m.is_evaluated]
    # get fitness values
    fitnesses = list(module.evaluate_values([m.value for m in unevaluated]))
    # save fitness values
    assert len(unevaluated) == len(fitnesses)
    for m, f in zip(unevaluated, fitnesses):
        m.fitness = f
    # return the count
    return len(unevaluated)


# ========================================================================= #
# Functional Trainer                                                        #
# ========================================================================= #


def yield_population_steps(module: EaModule[T]) -> Iterator[Tuple[int, Population[T], Population[T], int]]:
    # 1. create population
    population = [Member(m) for m in module.gen_starting_values()]
    population_size = len(population)
    population = _check_population(population, required_size=population_size)

    # 2. evaluate population
    evals = _evaluate_unevaluated(module, population)

    # yield initial population
    yield 0, population, population, evals

    # training loop
    for i in itertools.count(1):
        # 1. generate offspring
        offspring = module.generate_offspring(population)
        # 2. evaluate
        evals = _evaluate_unevaluated(module, offspring)
        # 3. select
        population = module.select_population(population, offspring)
        population = _check_population(population, required_size=population_size)

        # yield steps
        yield i, population, offspring, evals


# ========================================================================= #
# Class Trainer                                                             #
# ========================================================================= #


class Trainer(Generic[T]):

    def __init__(
        self,
        generations: int = 100,
        progress: bool = True,
        history_n_best: int = 5,
        offspring_generator=yield_population_steps,
    ):
        self._generations = generations
        self._progress = progress
        self._history_n_best = history_n_best
        self._offspring_generator = offspring_generator
        assert self._history_n_best > 0

    def fit(self, module: EaModule[T]) -> Tuple[Population[T], Logbook[T], HallOfFame[T]]:
        assert isinstance(module, EaModule)
        # history trackers
        logbook, halloffame = self._create_default_trackers(module)
        # progress bar and training loop
        with tqdm(total=self._generations, desc='generation', disable=not self._progress, ncols=120) as p:
            for gen, population, offspring, evals in itertools.islice(self._offspring_generator(module), self._generations):
                # update statistics with new population
                halloffame.update(offspring)
                stats = logbook.record(population, gen=gen, evals=evals)
                # update progress bar
                p.update()
                p.set_postfix({k: stats[k] for k in module.get_progress_stats()})
        # done
        return population, logbook, halloffame.freeze()

    def _create_default_trackers(self, module: EaModule[T]) -> Tuple[Logbook[T], HallOfFame[T]]:
        halloffame = HallOfFame(
            n_best=self._history_n_best,
            maximize=True,
        )
        logbook = Logbook(
            'gen', 'evals',
            fit=StatsGroup(lambda pop: [m.fitness for m in pop], min=np.min, max=np.max, mean=np.mean),
            **module.get_stats_groups()
        )
        return logbook, halloffame


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
