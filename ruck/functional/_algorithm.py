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
from typing import Optional
from typing import Tuple
from typing import TypeVar

import numpy as np

from ruck._member import Member
from ruck._member import Population
from ruck.functional import SelectFnHint
from ruck.functional._mate import MateFnHint
from ruck.functional._mutate import MutateFnHint
from ruck.util._iter import chained


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


T = TypeVar('T')


# ========================================================================= #
# Function Wrappers                                                         #
# ========================================================================= #


def apply_mate(
    population: Population[T],
    mate_fn: MateFnHint[T],
    p: float = 0.5,
    map_fn=map,
) -> Population[T]:
    # randomize order so we have randomized pairs
    offspring = list(population)
    np.random.shuffle(offspring)
    # select random items
    idxs, pairs = [], []
    for i, (m0, m1) in enumerate(zip(offspring[0::2], offspring[1::2])):
        if random.random() < p:
            pairs.append((m0.value, m1.value))
            idxs.append(i)
    # map selected values
    pairs = map_fn(lambda pair: mate_fn(pair[0], pair[1]), pairs)
    # update values
    for i, (v0, v1) in zip(idxs, pairs):
        offspring[i*2+0] = Member(v0)
        offspring[i*2+1] = Member(v1)
    # done!
    return offspring


def apply_mutate(
    population: Population,
    mutate_fn: MutateFnHint,
    p: float = 0.5,
    map_fn=map,
) -> Population:
    # shallow copy because we want to update elements in this list
    offspring = list(population)
    # select random items
    idxs, vals = [], []
    for i, m in enumerate(offspring):
        if random.random() < p:
            vals.append(m.value)
            idxs.append(i)
    # map selected values
    vals = map_fn(mutate_fn, vals)
    # update values
    for i, v in zip(idxs, vals):
        offspring[i] = Member(v)
    # done!
    return offspring


def apply_mate_and_mutate(
    population: Population[T],
    mate_fn: MateFnHint[T],
    mutate_fn: MutateFnHint[T],
    p_mate: float = 0.5,
    p_mutate: float = 0.5,
    map_fn=map,
) -> Population[T]:
    """
    Apply crossover AND mutation

    NOTE:
    - Modified individuals need their fitness re-evaluated
    - Mate & Mutate should always return copies of the received values.

    ** Should be equivalent to varAnd from DEAP **
    """
    offspring = apply_mate(population, mate_fn, p=p_mate, map_fn=map_fn)
    offspring = apply_mutate(offspring, mutate_fn, p=p_mutate, map_fn=map_fn)
    return offspring


def _get_generate_member_fn(
    mate_fn: MateFnHint[T],
    mutate_fn: MutateFnHint[T],
    p_mate: float = 0.5,
    p_mutate: float = 0.5,
):
    def _generate_member(a_b_r: Tuple[Member[T], Optional[Member[T]], float]) -> Member[T]:
        ma, mb, r = a_b_r
        if   r < p_mate:            return Member(mate_fn(ma.value, mb.value)[0])  # Apply crossover | only take first item | mb is only defined for this case
        elif r < p_mate + p_mutate: return Member(mutate_fn(ma.value))             # Apply mutation
        else:                       return ma                                      # Apply reproduction
    return _generate_member


def apply_mate_or_mutate_or_reproduce(
    population: Population[T],
    num_offspring: int,  # lambda_
    mate_fn: MateFnHint[T],
    mutate_fn: MutateFnHint[T],
    p_mate: float = 0.4,
    p_mutate: float = 0.4,
    map_fn=map,
) -> Population[T]:
    """
    Apply crossover OR mutation OR reproduction

    NOTE:
    - Modified individuals need their fitness re-evaluated
    - Mate & Mutate should always return copies of the received values.

    ** Should be equivalent to varOr from DEAP, but significantly faster for larger populations **
    """
    assert (p_mate + p_mutate) <= 1.0, 'The sum of the crossover and mutation probabilities must be smaller or equal to 1.0.'

    # get the actions to be performed
    # - the multinomial distribution models the numbers of
    #   times a specific outcome occurred after n trials
    num_mate, num_mutate, num_reproduce = np.random.multinomial(num_offspring, [p_mate, p_mutate, 1-(p_mate+p_mutate)])
    # randomly sample the offspring
    offspring_mate         = [random.choice(population)    for _ in range(num_mate)]
    offspring_pairs_mutate = [random.sample(population, 2) for _ in range(num_mutate)]
    offspring_reproduce    = [random.choice(population)    for _ in range(num_reproduce)]
    # apply the mating and mutations
    offspring_mate         = map_fn(mutate_fn, (m.value for m in offspring_mate))
    offspring_pairs_mutate = map_fn(lambda pair: mate_fn(pair[0], pair[1]), ((m0.value, m1.value) for m0, m1 in offspring_pairs_mutate))
    # combine everything & shuffle
    offspring = chained([
        (Member(v) for v in offspring_mate),
        (Member(v0) for v0, v1 in offspring_pairs_mutate),
        offspring_reproduce
    ])
    np.random.shuffle(offspring)
    # done!
    assert len(offspring) == num_offspring
    return offspring


# ========================================================================= #
# Gen & Select                                                              #
# ========================================================================= #


def make_ea(
    mate_fn: MateFnHint[T],
    mutate_fn: MutateFnHint[T],
    select_fn: SelectFnHint[T],
    offspring_num: int = None,   # lambda
    mode: str = 'simple',
    p_mate: float = 0.5,
    p_mutate: float = 0.5,
    map_fn=map,
):
    if offspring_num is not None:
        assert offspring_num > 0, f'invalid arguments, the number of offspring: {repr(offspring_num)} (lambda) must be > 0'

    if mode == 'simple':
        def generate(population):
            num = len(population) if (offspring_num is None) else offspring_num
            assert num == len(population), f'invalid arguments for mode={repr(mode)}, the number of offspring: {num} (lambda) must be equal to the size of the population: {len(population)} (mu)'
            offspring = apply_mate_and_mutate(population=select_fn(population, len(population)), p_mate=p_mate, mate_fn=mate_fn, p_mutate=p_mutate, mutate_fn=mutate_fn, map_fn=map_fn)
            return offspring

        def select(population, offspring):
            return offspring

    elif mode == 'mu_plus_lambda':
        def generate(population):
            num = len(population) if (offspring_num is None) else offspring_num
            return apply_mate_or_mutate_or_reproduce(population, num, mate_fn=mate_fn, mutate_fn=mutate_fn, p_mate=p_mate, p_mutate=p_mutate, map_fn=map_fn)

        def select(population: Population[T], offspring: Population[T]):
            return select_fn(population + offspring, len(population))

    elif mode == 'mu_comma_lambda':
        def generate(population):
            num = len(population) if (offspring_num is None) else offspring_num
            assert num >= len(population), f'invalid arguments for mode={repr(mode)}, the number of offspring: {num} (lambda) must be greater than or equal to the size of the population: {len(population)} (mu)'
            return apply_mate_or_mutate_or_reproduce(population, num, mate_fn=mate_fn, mutate_fn=mutate_fn, p_mate=p_mate, p_mutate=p_mutate, map_fn=map_fn)

        def select(population, offspring):
            return select_fn(offspring, len(population))

    else:
        raise KeyError(f'invalid mode: {repr(mode)}, must be one of: ["simple", "mu_plus_lambda", "mu_comma_lambda"]')

    return generate, select


# # ========================================================================= #
# # END                                                                       #
# # ========================================================================= #
