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

import warnings
from typing import Optional
from typing import Sequence

from ruck.functional import check_selection


try:
    import deap
except ImportError as e:
    warnings.warn('failed to import deap, please install it: $ pip install deap')
    raise e


# ========================================================================= #
# deap helper                                                               #
# ========================================================================= #


# CONFIG:            | JIT:                                                | PYTHON:
# (in=0008 out=0004) | [OLD:  0.000176 NEW: 0.000156s SPEEDUP:  1.126655x] | [OLD:  0.000139 NEW:  0.000198s SPEEDUP: 0.699888x]
# (in=0064 out=0032) | [OLD:  0.002818 NEW: 0.000316s SPEEDUP:  8.913371x] | [OLD:  0.002732 NEW:  0.003151s SPEEDUP: 0.867194x]
# (in=0256 out=0128) | [OLD:  0.040459 NEW: 0.001258s SPEEDUP: 32.161621x] | [OLD:  0.038630 NEW:  0.045156s SPEEDUP: 0.855490x]
# (in=1024 out=0512) | [OLD:  0.672029 NEW: 0.010862s SPEEDUP: 61.872225x] | [OLD:  0.644428 NEW:  0.768074s SPEEDUP: 0.839018x]
# (in=4096 out=2048) | [OLD: 10.511867 NEW: 0.158704s SPEEDUP: 66.235660x] | [OLD: 10.326754 NEW: 12.973584s SPEEDUP: 0.795983x]


@check_selection
def select_nsga2(population, num_offspring: int, weights: Optional[Sequence[float]] = None):
    """
    This is hacky... ruck doesn't yet have NSGA2
    support, but we will add it in future!
    """
    # this function has been deprecated
    warnings.warn('`ruck.external.deap.select_nsga2` has been deprecated in favour of `ruck.functional.select_nsga2`. `ruck.external.deap` will be removed in version v0.3.0')
    # checks
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
# END                                                                       #
# ========================================================================= #
