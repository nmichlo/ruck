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
import timeit
from collections import defaultdict
from itertools import chain
from pprint import pprint
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numba
import numpy as np
from deap.tools import sortNondominated
from deap.tools.emo import assignCrowdingDist

from ruck import Member
from ruck import Population
from ruck.functional import check_selection
from ruck.util.array import arggroup


try:
    import deap
except ImportError as e:
    import warnings
    warnings.warn('failed to import deap, please install it: $ pip install deap')
    raise e

# ========================================================================= #
# deap helper                                                               #
# ========================================================================= #


@check_selection
def select_nsga2(population, num_offspring: int, weights: Optional[Sequence[float]] = None):
    """
    This is hacky... ruck doesn't yet have NSGA2
    support, but we will add it in future!
    """
    if num_offspring == 0:
        return []
    # get a fitness value to perform checks
    f = population[0].fitness
    # check fitness
    try:
        for _ in f: break
    except:
        raise ValueError('fitness values do not have multiple values!')
    # get weights
    if weights is None:
        weights = tuple(1.0 for _ in f)
    # get deap
    from deap import creator, tools, base
    # initialize creator
    creator.create('_SelIdxFitness', base.Fitness, weights=weights)
    creator.create('_SelIdxIndividual', int, fitness=creator._SelIdxFitness)
    # convert to deap population
    idx_individuals = []
    for i, m in enumerate(population):
        ind = creator._SelIdxIndividual(i)
        ind.fitness.values = m.fitness
        idx_individuals.append(ind)
    # run nsga2
    chosen_idx = tools.selNSGA2(individuals=idx_individuals, k=num_offspring)
    # return values
    return [population[i] for i in chosen_idx]


# ========================================================================= #
# CUSTOM                                                                    #
# ========================================================================= #

def _get_fitnesses(population: Population, weights=None):
    fitnesses = np.array([m.fitness for m in population])
    # check dims
    if fitnesses.ndim == 1:
        fitnesses = fitnesses[:, None]
    assert fitnesses.ndim == 2
    # handle weights
    if weights is not None:
        weights = np.array(weights)
        # check dims
        if weights.ndim == 0:
            weights = weights[None]
        assert weights.ndim == 1
        # multiply
        fitnesses *= weights[None, :]
    # done!
    return fitnesses


def select_nsga2_custom(population: 'Population', num: int, weights: Sequence[float]):
    w_fitnesses = _get_fitnesses(population, weights)
    # apply non-dominated sorting & get groups of member indices
    pareto_fronts = sort_non_dominated(w_fitnesses, num)
    # choose all but the last group
    chosen = list(chain(*pareto_fronts[:-1]))
    # check if we need more elements
    missing = num - len(chosen)
    assert missing >= 0, 'This is a bug!'
    # add missing elements
    if missing > 0:
        dists = get_crowding_distances([population[i].fitness for i in pareto_fronts[-1]])
        idxs = np.argsort(-np.array(dists))  # negate instead of reverse with [::-1] to match sorted(..., reverse=True)
        chosen.extend(pareto_fronts[-1][i] for i in idxs[:missing])
    # return the original members
    return [population[i] for i in chosen]


def sort_non_dominated(w_fitnesses: Sequence[Tuple[float, ...]], num: int, first_front_only=False) -> list:
    if num == 0:
        return []
    # collect all the non-unique elements into groups
    unique, inverse, counts = np.unique(w_fitnesses, return_inverse=True, return_counts=True, axis=0)
    # sort everything
    u_fronts_idxs = _sort_non_dominated_unique(unique_values=unique, unique_counts=counts, num=num, pareto_front_only=first_front_only)
    # return indices
    return u_fronts_idxs


@numba.njit()
def _sort_non_dominated_unique(unique_values: np.ndarray, unique_counts: np.ndarray, num: int, pareto_front_only=False) -> list:
    """
    Sort the first *k* *individuals* into different nondomination levels
    using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
    see [Deb2002]_. This algorithm has a time complexity of :math:`O(MN^2)`,
    where :math:`M` is the number of objectives and :math:`N` the number of
    individuals.
    """
    assert unique_values.ndim >= 1
    assert unique_counts.ndim == 1
    assert len(unique_values) == len(unique_counts)

    # get the number of items still to be
    # selected so we can exit early
    N = min(int(np.sum(unique_counts)), num)
    U = len(unique_values)

    num_selected = 0
    prev_front = []
    fronts_idxs = [prev_front]

    dominated_count = np.zeros(U, dtype='int')
    dominated_matrix = np.zeros((U, U), dtype='bool')

    # get the first pareto optimal front as all the fitness values
    # that are not dominated by any of the other fitness values
    for i, i_fit in enumerate(unique_values):
        # check if a fitness value is dominated
        # by any of the other fitness values
        for j, j_fit in zip(range(i+1, U), unique_values[i+1:]):
            # check domination
            if dominates_numpy(i_fit, j_fit):
                dominated_count[j] += 1
                dominated_matrix[i, j] = True
            elif dominates_numpy(j_fit, i_fit):
                dominated_count[i] += 1
                dominated_matrix[j, i] = True
        # add to the front the fitness value
        # if it is not dominated by anything else
        if dominated_count[i] == 0:
            num_selected += unique_counts[i]
            prev_front.append(i)
            # exit early
            if num_selected >= N:
                return fronts_idxs

    # exit early
    if pareto_front_only:
        return fronts_idxs

    # repeatedly add the next fronts by checking which values
    # are dominated by the previous fronts and then removing the dominated statistics.
    while True:
        # add a new front
        next_front = []
        fronts_idxs.append(next_front)

        # todo this can be improved
        for i in prev_front:
            (js,) = np.where(dominated_matrix[i])
            for j in js:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    num_selected += unique_counts[j]
                    next_front.append(j)
                    # exit early
                    if num_selected >= N:
                        return fronts_idxs

        # update the last front
        prev_front = next_front


def get_crowding_distances(fitnesses: Sequence[Tuple[float, ...]]):
    """
    Assign a crowding distance to each individual's fitness. The
    crowding distance can be retrieve via the :attr:`crowding_dist`
    attribute of each individual's fitness.
    """
    if len(fitnesses) == 0:
        return []

    distances = [0.0] * len(fitnesses)
    crowd = [(f, i) for i, f in enumerate(fitnesses)]

    nobj = len(fitnesses[0])

    for i in range(nobj):
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

    return distances


@numba.njit()
def dominates_numpy(weighted_fitness: np.ndarray, weighted_fitness_other: np.ndarray):
    """
    Return true if ALL values are [greater than or equal] AND
    at least one value is strictly [greater than].
    - If any value is [less than] then it is non-dominating
    """
    # pre-multiply fitness values by weight
    return not np.any(weighted_fitness < weighted_fitness_other) and np.any(weighted_fitness > weighted_fitness_other)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
