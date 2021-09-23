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

from ruck._member import Member
from ruck._member import PopulationHint
from ruck.functional._mate import MateFnHint
from ruck.functional._mutate import MutateFnHint


import random
import numpy as np


# ========================================================================= #
# Crossover & Mutate Helpers                                                #
# ========================================================================= #


def apply_mate(
    population: PopulationHint,
    mate_fn: MateFnHint,
    p: float = 0.5,
) -> PopulationHint:
    # randomize order so we have randomized pairs
    offspring = list(population)
    np.random.shuffle(offspring)
    # apply mating to population -- why is this faster than pre-generating the boolean mask?
    for i in range(1, len(population), 2):
        if random.random() < p:
            v0, v1 = mate_fn(offspring[i-1].value, offspring[i].value)
            offspring[i-1], offspring[i] = Member(v0), Member(v1)
    # done!
    return offspring


def apply_mutate(
    population: PopulationHint,
    mutate_fn: MutateFnHint,
    p: float = 0.5,
) -> PopulationHint:
    elem_mask = np.random.random(size=len(population)) < p
    # apply mutate to population
    return [
        Member(mutate_fn(m.value)) if do_mutate else m
        for m, do_mutate in zip(population, elem_mask)
    ]


def apply_mate_and_mutate(
    population: PopulationHint,
    mate_fn: MateFnHint,
    mutate_fn: MutateFnHint,
    p_mate: float = 0.5,
    p_mutate: float = 0.5,
) -> PopulationHint:
    """
    Apply crossover AND mutation

    NOTE:
    - Modified individuals need their fitness re-evaluated
    - Mate & Mutate should always return copies of the received values.

    ** Should be equivalent to varAnd from DEAP **
    """
    population = apply_mate(population, mate_fn, p=p_mate)
    population = apply_mutate(population, mutate_fn, p=p_mutate)
    return population


def apply_mate_or_mutate_or_reproduce(
    population: PopulationHint,
    num_offspring: int,  # lambda_
    mate_fn: MateFnHint,
    mutate_fn: MutateFnHint,
    p_mate: float = 0.5,
    p_mutate: float = 0.5,
) -> PopulationHint:
    """
    Apply crossover OR mutation OR reproduction

    NOTE:
    - Modified individuals need their fitness re-evaluated
    - Mate & Mutate should always return copies of the received values.

    ** Should be equivalent to varOr from DEAP, but significantly faster for larger populations **
    """
    assert (p_mate + p_mutate) <= 1.0, 'The sum of the crossover and mutation probabilities must be smaller or equal to 1.0.'

    pairs = np.random.randint(0, len(population), size=[2, num_offspring])
    rand  = np.random.random(len(population))

    def _fn(a: int, b: int, r: float):
        if   r < p_mate:            return Member(mate_fn(population[a].value, population[b].value)[0])  # Apply crossover
        elif r < p_mate + p_mutate: return Member(mutate_fn(population[a].value))  # Apply mutation
        else:                       return population[a]  # Apply reproduction

    # np.vectorize can help, but only about 10% faster for large populations, and 3x slower for tiny populations
    return [_fn(a, b, r) for a, b, r in zip(pairs[0], pairs[1], rand)]


# ========================================================================= #
# Gen & Select                                                              #
# ========================================================================= #


# def factory_ea_alg(
#     mate_fn,
#     mutate_fn,
#     select_fn,
#     mode: str = 'simple',
#     p_mate: float = 0.5,
#     p_mutate: float = 0.5,
#     offspring_num: int = 128,
#     population_num: int = 128,
# ):
#     if mode == 'simple':
#         def _generate(population):          return apply_mate_and_mutate(population=select_fn(population, len(population)), p_mate=p_mate, mate=mate_fn, p_mutate=p_mutate, mutate=mutate_fn)
#         def _select(population, offspring): return offspring
#     elif mode == 'mu_plus_lambda':
#         def _generate(population):          return apply_mate_or_mutate_or_reproduce(population, num_offspring=offspring_num, p_mate=p_mate, mate=mate_fn, p_mutate=p_mutate, mutate=mutate_fn)
#         def _select(population, offspring): return select_fn(population + offspring, population_num)
#     elif mode == 'mu_comma_lambda':
#         def _generate(population):          return apply_mate_or_mutate_or_reproduce(population, num_offspring=offspring_num, p_mate=p_mate, mate=mate_fn, p_mutate=p_mutate, mutate=mutate_fn)
#         def _select(population, offspring): return select_fn(offspring, population_num)
#     else:
#         raise KeyError(f'invalid mode: {repr(mode)}')
#     return _generate, _select


# # ========================================================================= #
# # END                                                                       #
# # ========================================================================= #
