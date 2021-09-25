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


import functools
import random
import numpy as np
import pytest

from examples.onemax import OneMaxModule
from examples.onemax_minimal import OneMaxMinimalModule
from ruck import Member
from ruck import Trainer
from ruck import R


# ========================================================================= #
# TESTS                                                                     #
# ========================================================================= #


def test_mate_keep_order():
    random.seed(77)
    np.random.seed(77)
    # checks
    offspring = R.apply_mate(
        population=[Member(c) for c in 'abcde'],
        mate_fn=lambda a, b: (a.upper(), b.upper()),
        p=0.5,
        keep_order=True,
    )
    # done
    assert ''.join(m.value for m in offspring) == 'ABcde'


def test_mate_random_order():
    random.seed(77)
    np.random.seed(77)
    # checks
    offspring = R.apply_mate(
        population=[Member(c) for c in 'abcde'],
        mate_fn=lambda a, b: (a.upper(), b.upper()),
        p=0.5,
        keep_order=False,
    )
    # done
    assert ''.join(m.value for m in offspring) == 'cdBAe'


def test_onemax_minimal():
    module = OneMaxMinimalModule()
    pop, logbook, halloffame = Trainer(generations=40, progress=False).fit(module)
    assert logbook[0]['fit:max'] < logbook[-1]['fit:max']


def test_onemax():
    module = OneMaxModule(population_size=300, member_size=100)
    pop, logbook, halloffame = Trainer(generations=40, progress=False).fit(module)
    assert logbook[0]['fit:max'] < logbook[-1]['fit:max']


def test_onemax_ea_simple():
    module = OneMaxModule(population_size=300, member_size=100)

    # EA SIMPLE
    module.generate_offspring, module.select_population = R.factory_simple_ea(
        mate_fn=R.mate_crossover_1d,
        mutate_fn=functools.partial(R.mutate_flip_bit_groups, p=0.05),
        select_fn=functools.partial(R.select_tournament, k=3),
        p_mate=module.hparams.p_mate,
        p_mutate=module.hparams.p_mutate,
    )

    pop, logbook, halloffame = Trainer(generations=40, progress=False).fit(module)
    assert logbook[0]['fit:max'] < logbook[-1]['fit:max']


def test_onemax_mu_plus_lambda():
    module = OneMaxModule(population_size=300, member_size=100)

    # MU PLUS LAMBDA
    module.generate_offspring, module.select_population = R.factory_mu_plus_lambda(
        mate_fn=R.mate_crossover_1d,
        mutate_fn=functools.partial(R.mutate_flip_bit_groups, p=0.05),
        select_fn=functools.partial(R.select_tournament, k=3),
        offspring_num=250,
        p_mate=module.hparams.p_mate,
        p_mutate=module.hparams.p_mutate,
    )

    pop, logbook, halloffame = Trainer(generations=40, progress=False).fit(module)
    assert logbook[0]['fit:max'] < logbook[-1]['fit:max']


def test_onemax_mu_comma_lambda():
    module = OneMaxModule(population_size=300, member_size=100)

    # MU COMMA LAMBDA
    module.generate_offspring, module.select_population = R.factory_mu_comma_lambda(
        mate_fn=R.mate_crossover_1d,
        mutate_fn=functools.partial(R.mutate_flip_bit_groups, p=0.05),
        select_fn=functools.partial(R.select_tournament, k=3),
        offspring_num=250,  # INVALID
        p_mate=module.hparams.p_mate,
        p_mutate=module.hparams.p_mutate,
    )

    with pytest.raises(AssertionError, match=r'invalid arguments, the number of offspring: 250 \(lambda\) must be greater than or equal to the size of the population: 300 \(mu\)'):
        pop, logbook, halloffame = Trainer(generations=40, progress=False).fit(module)

    # MU COMMA LAMBDA
    module.generate_offspring, module.select_population = R.factory_mu_comma_lambda(
        mate_fn=R.mate_crossover_1d,
        mutate_fn=functools.partial(R.mutate_flip_bit_groups, p=0.05),
        select_fn=functools.partial(R.select_tournament, k=3),
        offspring_num=400,
        p_mate=module.hparams.p_mate,
        p_mutate=module.hparams.p_mutate,
    )

    pop, logbook, halloffame = Trainer(generations=40, progress=False).fit(module)
    assert logbook[0]['fit:max'] < logbook[-1]['fit:max']



def test_member():
    m = Member('abc')
    assert str(m) == "Member('abc')"
    m = Member('abc', 0.5)
    assert str(m) == "Member('abc', 0.5)"
    m = Member('abc'*100, 0.5)
    assert str(m) == "Member('abcabcabcabca ... cabcabcabcabc', 0.5)"
    m = Member('abc  '*100, 0.5)
    assert str(m) == "Member('abc abc abc a ... abc abc abc ', 0.5)"


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
