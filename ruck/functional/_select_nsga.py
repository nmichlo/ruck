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

from typing import List
from typing import Optional
from typing import Sequence

import numpy as np

from ruck._member import Population
from ruck.external._numba import optional_njit
from ruck.functional._select import check_selection
from ruck.util._array import arggroup
from ruck.util._population import population_fitnesses


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


@check_selection
def select_nsga2(population: Population, num: int, weights: Optional[Sequence[float]] = None):
    """
    NSGA-II works by:
        1. grouping the population fitness values into successive fronts with non-dominated sorting.
           - only the first F fronts that contain at least N elements are returned, such that if
             there were F-1 fronts we would have < N elements.
        2. filling in missing elements using the crowding distance.
           - The first F-1 fronts are used by default containing K elements. The
             remaining N - K elements are selected using the crowding distances between
             elements of the of the Fth front.

    ALGORITHM: according to the original NSGA-II paper
        "between two solutions with differing non-domination ranks we prefer the point
         with the lower rank. Otherwise, if both the points belong to the same front
         then we prefer the point which is located in a region with lesser number of
         points (the size of the cuboid inclosing it is larger)."

    CITATION:
        Deb, Kalyanmoy, et al. "A fast elitist non-dominated sorting genetic algorithm for
        multi-objective optimization: NSGA-II." International conference on parallel problem
        solving from nature. Springer, Berlin, Heidelberg, 2000.
    """
    # 1. apply non-dominated sorting to get the sequential non-dominated fronts
    fitnesses = population_fitnesses(population, weights=weights)
    fronts = argsort_non_dominated(fitnesses, at_least_n=num)
    # check if we need more elements
    chosen = [i for front in fronts[:-1] for i in front]
    missing = num - len(chosen)
    assert missing >= 0
    # 2. Add missing elements prioritising those with the largest crowding distances.
    #    Using weighted vs. original fitness values does not affect the crowding distance!
    #    NOTE: should this not operate over all elements? this feels weird just over the last front?
    if missing > 0:
        assert len(fronts[-1]) >= missing
        front_fitnesses = [population[i].fitness for i in fronts[-1]]  # TODO: `front_fitnesses = fitnesses[fronts[-1]]` should work, something is wrong with the `get_crowding_distances` function!
        # compute distances
        dists = compute_crowding_distances(front_fitnesses)
        idxs = np.argsort(-np.array(dists))
        chosen.extend(fronts[-1][i] for i in idxs[:num - len(chosen)])
    # return the original members
    assert len(chosen) == num
    return [population[i] for i in chosen]


# ========================================================================= #
# Non-Dominated Sorting                                                     #
# ========================================================================= #


def argsort_non_dominated(fitnesses: np.array, at_least_n: int = None, first_front_only=False) -> List[List[int]]:
    """
    Perform non-dominated arg-sorting on the elements in the array
    - The indices are contained in "fonts". Each front is a list of points
      that are not dominated by points in following fronts.
    - returns at least `at_least_n` indices, or all
      the indices if not specified.

    The algorithm works by repeatedly finding non-dominated fronts, and
    then removing them from the sets of points being considered.
    - An implementation detail is that we first check for duplicate
      points, which are then re-added into the fronts after they are found.

    NOTE: A more efficient version of the algorithm
          exists, but this is good enough for now.
    """
    if at_least_n is None:
        at_least_n = len(fitnesses)
    # checks
    assert at_least_n >= 0
    assert at_least_n <= len(fitnesses)
    # exit early
    if at_least_n == 0:
        return []
    # collect all the non-unique elements into groups
    groups, unique, counts = arggroup(fitnesses, keep_order=True, return_unique=True, return_counts=True, axis=0)
    # sort the unique elements into fronts
    u_fronts_idxs = _argsort_non_dominated_unique(
        unique_values=unique,
        unique_counts=counts,
        at_least_n=at_least_n,
        pareto_front_only=first_front_only,
    )
    # expand the groups back into fronts
    # that can contain repeated values
    return [[g for i in front for g in groups[i]] for front in u_fronts_idxs]


@optional_njit()
def _argsort_non_dominated_unique(
    unique_values: np.ndarray,
    unique_counts: np.ndarray,
    at_least_n: int,
    pareto_front_only: bool = False,
) -> list:
    assert unique_values.ndim == 2
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
            if _dominates(i_fit, j_fit):
                dominated_count[j] += 1
                dominates_lists[i].append(j)
            elif _dominates(j_fit, i_fit):
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
def _dominates(w_fitness, w_fitness_other):
    """
    Return true if ALL values are [greater than or equal] AND
    at least one value is strictly [greater than].
    - If any value is [less than] then it is non-dominating
    - both arrays must be one dimensional and the same size, we don't check for this!
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


# this is usually faster without JIT for very large arrays,
# but for small inputs the JIT is often better < 64
#
#   SHAPE: (N, F)     | JIT:                                             | NO-JIT:                                            #
# ┏ shape=(0, 2)      | OLD: 0.000001s NEW: 0.000002s SPEEDUP: 0.368500  | OLD: 0.000000s NEW: 0.000002s SPEEDUP: 0.202025  ┓ #
# ┗ shape=(0, 16)     | OLD: 0.000000s NEW: 0.000001s SPEEDUP: 0.266382  | OLD: 0.000000s NEW: 0.000001s SPEEDUP: 0.193062  ┛ #
# ┏ shape=(1, 2)      | OLD: 0.000004s NEW: 0.000001s SPEEDUP: 3.007124  | OLD: 0.000004s NEW: 0.000007s SPEEDUP: 0.578199  ┓ #
# ┗ shape=(1, 16)     | OLD: 0.000015s NEW: 0.000003s SPEEDUP: 5.763490  | OLD: 0.000015s NEW: 0.000036s SPEEDUP: 0.417297  ┛ #
# ┏ shape=(8, 2)      | OLD: 0.000015s NEW: 0.000002s SPEEDUP: 6.663598  | OLD: 0.000015s NEW: 0.000017s SPEEDUP: 0.885061  ┓ #
# ┗ shape=(8, 16)     | OLD: 0.000092s NEW: 0.000009s SPEEDUP: 10.708658 | OLD: 0.000091s NEW: 0.000103s SPEEDUP: 0.883329  ┛ #
# ┏ shape=(64, 2)     | OLD: 0.000099s NEW: 0.000004s SPEEDUP: 23.352296 | OLD: 0.000094s NEW: 0.000018s SPEEDUP: 5.085974  ┓ #
# ┗ shape=(64, 16)    | OLD: 0.000702s NEW: 0.000029s SPEEDUP: 24.185735 | OLD: 0.000670s NEW: 0.000121s SPEEDUP: 5.514704  ┛ #
# ┏ shape=(256, 2)    | OLD: 0.000394s NEW: 0.000013s SPEEDUP: 29.474941 | OLD: 0.000385s NEW: 0.000029s SPEEDUP: 13.172985 ┓ #
# ┗ shape=(256, 16)   | OLD: 0.002923s NEW: 0.000177s SPEEDUP: 16.466151 | OLD: 0.002780s NEW: 0.000252s SPEEDUP: 11.045761 ┛ #
# ┏ shape=(1024, 2)   | OLD: 0.001671s NEW: 0.000103s SPEEDUP: 16.259937 | OLD: 0.001584s NEW: 0.000107s SPEEDUP: 14.738836 ┓ #
# ┗ shape=(1024, 16)  | OLD: 0.012324s NEW: 0.000879s SPEEDUP: 14.016850 | OLD: 0.011636s NEW: 0.000838s SPEEDUP: 13.879243 ┛ #
# ┏ shape=(16384, 2)  | OLD: 0.029240s NEW: 0.002433s SPEEDUP: 12.016264 | OLD: 0.028420s NEW: 0.002033s SPEEDUP: 13.977902 ┓ #
# ┗ shape=(16384, 16) | OLD: 0.214716s NEW: 0.020512s SPEEDUP: 10.467601 | OLD: 0.212408s NEW: 0.016327s SPEEDUP: 13.009210 ┛ #

def compute_crowding_distances(positions) -> np.ndarray:
    """
    Compute the crowding distance for each position in an array.
    - These are usually a 2D array of fitness values

    The general algorithm for the crowding distance is:
    1. for each component of the fitness/position, add to the distance for each element:
    2.     | sort all the entries according to this component
    3.     | the endpoints of the sorted array are assigned infinite distance
    4.     | add to the distance for the middle element looking at consecutive triples,
           | the value added is the distance between its two neighbours over the normalisation
           | value, which is a multiple of the (max - min) over all values

    TODO: Is there not a better algorithm than this? The crowding
          distance seems very dependant on alignment with axes?
    """
    # make sure we have the right datatype
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions, dtype='float64')
    return _get_crowding_distances(positions)


@optional_njit()
def _get_crowding_distances(positions: np.ndarray) -> np.ndarray:
    # get shape
    assert positions.ndim == 2
    N, F = positions.shape
    # exit early
    if N == 0:
        return np.zeros(0, dtype='float64')
    # store for the values
    distances = np.zeros(N, dtype='float64')
    # 1. for each fitness component, update the distance for each member!
    for crowd in positions.T:
        # 2. sort in increasing order
        crowd_idxs = np.argsort(crowd)
        crowd = crowd[crowd_idxs]
        # 3. update endpoint distances
        distances[crowd_idxs[0]] = np.inf
        distances[crowd_idxs[-1]] = np.inf
        # get endpoints
        m = crowd[0]
        M = crowd[-1]
        # skip if the endpoints are the same (values will be zero if this is the case)
        # or if there are not enough elements to compute over consecutive triples
        if (M == m) or (len(crowd) < 3):
            continue
        # normalize the values between the maximum and minimum distances
        # NOTE: the original NSGA-II paper does not apply this normalization constant...
        norm = F * (M - m)
        # 4. compute the distance as the difference between the previous
        # and next values all over the normalize distance
        distances[crowd_idxs[1:-1]] += (crowd[2:] - crowd[:-2]) / norm
    # return the distances!
    return distances


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
