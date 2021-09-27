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
# END                                                                       #
# ========================================================================= #
