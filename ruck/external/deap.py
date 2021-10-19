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

from itertools import chain
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import front as front
import numpy as np

from ruck import Population
from ruck.external._numba import optional_njit
from ruck.functional import check_selection
from ruck.util._population import population_fitnesses


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
# Helper                                                                    #
# ========================================================================= #


def select_nsga2_custom(population: 'Population', num: int, weights: Sequence[float]):
    """
    NSGA-II works by:

        1. grouping the population fitness values into successive fronts with non-dominated sorting.
           - only the first F fronts that contain at least N elements are returned, such that if
             there were F-1 fronts we would have < N elements.

        2. filling in missing elements using the crowding distance.
           - The first F-1 fronts are used by default containing K elements. The
             remaining N - K elements are selected using the crowding distances between
             elements of the of the Fth front.

            TODO: this doesnt sound correct, surely you would include all elements not
                  in the first front when checking the crowding distances?
    """
    # apply non-dominated sorting and get progressive fronts
    fronts = argsort_non_dominated(population_fitnesses(population, weights=weights), at_least_n=num)

    # chain fronts together
    chosen = [i for front in fronts[:-1] for i in front]

    # check if we need more elements
    missing = num - len(chosen)
    assert missing >= 0, 'This is a bug!'

    # add missing elements according to crowding distance
    if missing > 0:
        dists = get_crowding_distances([population[i].fitness for i in fronts[-1]])  # TODO: is this correct, should it not be the below?
        idxs = np.argsort(-np.array(dists))  # negate instead of reverse with [::-1] to match sorted(..., reverse=True)
        chosen.extend(fronts[-1][i] for i in idxs[:missing])

    # checks
    assert len(chosen) == num, 'This is a bug!'
    # return the original members
    return [population[i] for i in chosen]


# ========================================================================= #
# Non-Dominated Sorting                                                     #
# ========================================================================= #


def arggroup(
    numbers: Union[Sequence, np.ndarray],
    axis=0,
    keep_order=True,
    return_unique: bool = False,
    return_index: bool = False,
    return_counts: bool = False,
):
    """
    Group all the elements of the array.
    - The returned groups contain the indices of
      the original position in the arrays.
    """

    # convert
    if not isinstance(numbers, np.ndarray):
        numbers = np.array(numbers)
    # checks
    if numbers.ndim == 0:
        raise ValueError('input array must have at least one dimension')
    if numbers.size == 0:
        return []
    # we need to obtain the sorted groups of
    unique, index, inverse, counts = np.unique(numbers, return_index=True, return_inverse=True, return_counts=True, axis=axis)
    # same as [ary[:idx[0]], ary[idx[0]:idx[1]], ..., ary[idx[-2]:idx[-1]], ary[idx[-1]:]]
    groups = np.split(ary=np.argsort(inverse, axis=0), indices_or_sections=np.cumsum(counts)[:-1], axis=0)
    # maintain original order
    if keep_order:
        add_order = index.argsort()  # the order that items were added in
        groups = [groups[i] for i in add_order]
    # return values
    results = [groups]
    if return_unique:  results.append(unique[add_order] if keep_order else unique)
    if return_index:   results.append(index[add_order]  if keep_order else index)
    if return_counts:  results.append(counts[add_order] if keep_order else counts)
    # unpack
    if len(results) == 1:
        return results[0]
    return results


def argsort_non_dominated(fitnesses: np.array, at_least_n: int = None, first_front_only=False) -> list:
    if at_least_n is None:
        at_least_n = len(fitnesses)
    elif at_least_n == 0:
        return []
    # collect all the non-unique elements into groups
    groups, unique, counts = arggroup(fitnesses, keep_order=True, return_unique=True, return_counts=True, axis=0)
    # sort into fronts
    u_fronts_idxs = _argsort_non_dominated_unique(
        unique_values=unique,
        unique_counts=counts,
        at_least_n=at_least_n,
        pareto_front_only=first_front_only,
    )
    return [[g for i in front for g in groups[i]] for front in u_fronts_idxs]



@optional_njit()
def _argsort_non_dominated_unique(
    unique_values: np.ndarray,
    unique_counts: np.ndarray,
    at_least_n: int,
    pareto_front_only: bool = False,
) -> list:
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
    N = min(int(np.sum(unique_counts)), at_least_n)
    U = len(unique_values)

    # exit early if we need to
    num_selected = 0

    # store the indices of the values that form the first front,
    # containing all the items that are not dominated by anything else!
    first_front = []

    # store the counts for each value based on
    # the number of elements that dominate it
    dominated_count = np.zeros(U, dtype='int')

    # construct a list of lists to store all the indices of the
    # items that the parent item dominates
    dominates_lists = []
    for _ in range(U):
        # numba cannot infer the types if we dont do this
        dominates_lists.append([-1])
        dominates_lists[-1].clear()

    # get the first pareto optimal front as all the fitness values
    # that are not dominated by any of the other fitness values
    for i, i_fit in enumerate(unique_values):
        # check if a fitness value is dominated by any of the other fitness values
        # and update its statistics if it is
        for j, j_fit in zip(range(i+1, U), unique_values[i + 1:]):
            if dominates(i_fit, j_fit):
                dominated_count[j] += 1
                dominates_lists[i].append(j)
            elif dominates(j_fit, i_fit):
                dominated_count[i] += 1
                dominates_lists[j].append(i)

        # add to the front the fitness value
        # if it is not dominated by anything else
        if dominated_count[i] == 0:
            num_selected += unique_counts[i]
            first_front.append(i)

    # exit early
    if pareto_front_only or (num_selected >= N):
        return [first_front]

    # create the remaining fronts
    prev_front = first_front
    fronts = [first_front]

    # repeatedly generate the next fronts
    while True:
        # add a new front
        curr_front = []
        fronts.append(curr_front)

        # for each item in a previous front
        added = False
        for i in prev_front:
            # check all the items that the previous value dominates
            for j in dominates_lists[i]:
                dominated_count[j] -= 1
                # add the item to the current front if it is only dominated
                # by elements in the fronts before it.
                if dominated_count[j] == 0:
                    num_selected += unique_counts[j]
                    curr_front.append(j)
                    added = True

        # make sure that there is not an infinite
        # loop if there is accidentally a bug!
        if not added:
            raise RuntimeError('This is a bug!')

        # exit early
        if num_selected >= N:
            return fronts

        # update the previous front
        prev_front = curr_front


@optional_njit()
def dominates(w_fitness, w_fitness_other):
    """
    Return true if ALL values are [greater than or equal] AND
    at least one value is strictly [greater than].
    - If any value is [less than] then it is non-dominating
    """
    dominated = False
    for iv, jv in zip(w_fitness, w_fitness_other):
        if iv > jv:
            dominated = True
        elif iv < jv:
            return False
    return dominated


# ========================================================================= #
# CROWDING DISTANCES                                                        #
# ========================================================================= #


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


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
