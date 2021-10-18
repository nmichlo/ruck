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

from typing import Optional
from typing import Tuple
from ruck.functional import check_selection


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
def select_nsga2(population, num_offspring: int, weights: Optional[Tuple[float, ...]] = None):
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


def select_nsga2_custom(population: 'Population', num: int, weights: np.ndarray):
    pareto_fronts = sort_non_dominated(population, num, weights)
    chosen = list(chain(*pareto_fronts[:-1]))
    k = num - len(chosen)
    if k > 0:
        front = pareto_fronts[-1]
        dists = get_crowding_distances(front)
        idxs = np.argsort(-np.array(dists))
        chosen.extend(front[i] for i in idxs[:k])
    return chosen


def sort_non_dominated(population: Population, num: int, weights: np.ndarray, first_front_only=False) -> list:
    """
    Sort the first *k* *individuals* into different nondomination levels
    using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
    see [Deb2002]_. This algorithm has a time complexity of :math:`O(MN^2)`,
    where :math:`M` is the number of objectives and :math:`N` the number of
    individuals.
    """
    if num == 0:
        return []

    map_fit_ind = defaultdict(list)
    for member in population:
        map_fit_ind[member.fitness].append(member)
    fits = list(map_fit_ind.keys())

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    # Rank first Pareto front
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i+1:]:
            if dominates(fit_i, fit_j, weights):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif dominates(fit_j, fit_i, weights):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    # Rank the next front until all individuals are sorted or
    # the given number of individual are sorted.
    if not first_front_only:
        N = min(len(population), num)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts


def get_crowding_distances(population: Population):
    """
    Assign a crowding distance to each individual's fitness. The
    crowding distance can be retrieve via the :attr:`crowding_dist`
    attribute of each individual's fitness.
    """
    if len(population) == 0:
        return []

    distances = [0.0] * len(population)
    crowd = [(member.fitness, i) for i, member in enumerate(population)]

    nobj = len(population[0].fitness)

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


def dominates(fitness: Sequence[float], fitness_other: Sequence[float], weights: Sequence[float]):
    """
    Return true if ALL values are [greater than or equal] AND
    at least one value is strictly [greater than].
    - If any value is [less than] then it is non-dominating

    == `all_greater_than_or_equal and any_greater_than`
    == `(not any_less_than and any_greater_than)`                 *NB*
    == `(not any_less_than and not all_less_than_or_equal)`
    == `not (any_less_than or all_less_than_or_equal)`
    == `not (any_less_than or all_equal)`
    """
    num_greater = 0
    for fit_0, fit_1, weight in zip(fitness, fitness_other, weights):
        w_fit_0 = weight * fit_0
        w_fit_1 = weight * fit_1
        # update equal
        if w_fit_0 > w_fit_1:
            num_greater += 1
        if w_fit_0 < w_fit_1:
            # we exit early if any value is [less than]
            return False
    # We return true if all the values are  [greater than or equal]
    # AND at least one is strictly [greater than]
    return num_greater > 0


def dominates_numpy(fitness: np.ndarray, fitness_other: np.ndarray, weights: np.ndarray):
    f0 = weights * fitness
    f1 = weights * fitness_other
    return not np.any(f0 < f1) and np.any(f0 > f1)

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
