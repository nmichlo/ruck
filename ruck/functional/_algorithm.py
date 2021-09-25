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
from typing import TypeVar

from ruck._member import Member
from ruck._member import Population
from ruck.functional import SelectFnHint
from ruck.functional._mate import MateFnHint
from ruck.functional._mutate import MutateFnHint
from ruck.util._iter import replaced_random_taken_pairs
from ruck.util._iter import replaced_random_taken_elems


import random
import numpy as np


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


T = TypeVar('T')


# ========================================================================= #
# Crossover & Mutate Helpers                                                #
# ========================================================================= #


def _mate_wrap_unwrap_values(mate_fn: MateFnHint[T]):
    def wrapper(ma: Member[T], mb: Member[T]) -> Tuple[Member[T], Member[T]]:
        va, vb = mate_fn(ma.value, mb.value)
        return Member(va), Member(vb)
    return wrapper


def _mutate_wrap_unwrap_values(mutate_fn: MutateFnHint[T]):
    def wrapper(m: Member[T]) -> Member[T]:
        v = mutate_fn(m.value)
        return Member(v)
    return wrapper


# ========================================================================= #
# Function Wrappers                                                         #
# ========================================================================= #


def apply_mate(
    population: Population[T],
    mate_fn: MateFnHint[T],
    p: float = 0.5,
    keep_order: bool = True,
    map_fn=map,
) -> Population[T]:
    # randomize order so we have randomized pairs
    if keep_order:
        indices = np.arange(len(population))
        np.random.shuffle(indices)
        offspring = [population[i] for i in indices]
    else:
        offspring = list(population)
        np.random.shuffle(offspring)
    # apply mating to population
    offspring = replaced_random_taken_pairs(
        fn=_mate_wrap_unwrap_values(mate_fn),
        items=offspring,
        p=p,
        map_fn=map_fn,
    )
    # undo random order
    if keep_order:
        offspring = [offspring[i] for i in np.argsort(indices)]
    # done!
    assert len(offspring) == len(population)
    return offspring


def apply_mutate(
    population: Population,
    mutate_fn: MutateFnHint,
    p: float = 0.5,
    map_fn=map,
) -> Population:
    # apply mutations to population
    offspring = replaced_random_taken_elems(
        fn=_mutate_wrap_unwrap_values(mutate_fn),
        items=population,
        p=p,
        map_fn=map_fn,
    )
    # done!
    assert len(offspring) == len(population)
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
    offspring = apply_mate(population, mate_fn, p=p_mate, keep_order=True, map_fn=map_fn)
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
    p_mate: float = 0.5,
    p_mutate: float = 0.5,
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

    # choose which action should be taken for each element
    probabilities  = np.random.random(num_offspring)
    # select offspring
    choices_a = [random.choice(population)                           for p in probabilities]
    choices_b = [random.choice(population) if (p < p_mate) else None for p in probabilities]  # these are only needed for crossover, when (p < p_mate)
    # get function to generate offspring
    # - we create the function so that we don't accidentally pickle anything else
    fn = _get_generate_member_fn(mate_fn=mate_fn, mutate_fn=mutate_fn, p_mate=p_mate, p_mutate=p_mutate)
    # generate offspring
    # - TODO: this is actually not optimal! we should only pass mate and
    #         mutate operations to the map function, we could distribute
    #         work unevenly between processes if map_fn is replaced
    offspring = list(map_fn(fn, zip(choices_a, choices_b, probabilities)))
    # done!
    assert len(offspring) == num_offspring
    return offspring


# ========================================================================= #
# Gen & Select                                                              #
# ========================================================================= #


def factory_simple_ea(
    mate_fn: MateFnHint[T],
    mutate_fn: MutateFnHint[T],
    select_fn: SelectFnHint[T],
    p_mate: float = 0.5,
    p_mutate: float = 0.5,
    map_fn=map,
):
    def generate(population):
        return apply_mate_and_mutate(population=select_fn(population, len(population)), p_mate=p_mate, mate_fn=mate_fn, p_mutate=p_mutate, mutate_fn=mutate_fn, map_fn=map_fn)

    def select(population, offspring):
        return offspring

    return generate, select


def factory_mu_plus_lambda(
    mate_fn: MateFnHint[T],
    mutate_fn: MutateFnHint[T],
    select_fn: SelectFnHint[T],
    offspring_num: int,   # lambda
    p_mate: float = 0.5,
    p_mutate: float = 0.5,
    map_fn=map,
):
    def generate(population):
        num = len(population) if (offspring_num is None) else offspring_num
        return apply_mate_or_mutate_or_reproduce(population, num, mate_fn=mate_fn, mutate_fn=mutate_fn, p_mate=p_mate, p_mutate=p_mutate, map_fn=map_fn)

    def select(population: Population[T], offspring: Population[T]):
        return select_fn(population + offspring, len(population))

    return generate, select


def factory_mu_comma_lambda(
    mate_fn: MateFnHint[T],
    mutate_fn: MutateFnHint[T],
    select_fn: SelectFnHint[T],
    offspring_num: Optional[int] = None, # lambda
    p_mate: float = 0.5,
    p_mutate: float = 0.5,
    map_fn=map,
):
    def generate(population):
        num = len(population) if (offspring_num is None) else offspring_num
        return apply_mate_or_mutate_or_reproduce(population, num, mate_fn=mate_fn, mutate_fn=mutate_fn, p_mate=p_mate, p_mutate=p_mutate, map_fn=map_fn)

    def select(population, offspring):
        assert len(offspring) >= len(population), f'invalid arguments, the number of offspring: {len(offspring)} (lambda) must be greater than or equal to the size of the population: {len(population)} (mu)'
        return select_fn(offspring, len(population))

    return generate, select


# # ========================================================================= #
# # END                                                                       #
# # ========================================================================= #
