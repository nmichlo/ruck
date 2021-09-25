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

import numpy as np
import pytest

from examples.onemax import OneMaxModule
from examples.onemax_ray import OneMaxRayModule
from ruck import Member
from ruck import Trainer
from ruck import R


# ========================================================================= #
# TESTS                                                                     #
# ========================================================================= #


def test_mate_random_order():
    random.seed(77)
    np.random.seed(77)
    # checks
    offspring = R.apply_mate(
        population=[Member(c) for c in 'abcde'],
        mate_fn=lambda a, b: (a.upper(), b.upper()),
        p=0.5,
    )
    # done
    assert ''.join(m.value for m in offspring) == 'cdBAe'


@pytest.mark.parametrize(['module_cls', 'generations', 'ea_mode', 'population_size', 'offspring_num', 'assertion_error'], [
    # advanced module
    (OneMaxModule,   40, 'simple',          200, None, None),
    # (OneMaxModule, 40, 'simple',          200,   0, r"invalid arguments, the number of offspring: 0 \(lambda\) must be > 0"),
    (OneMaxModule,   40, 'simple',          200, 150, r"invalid arguments for mode='simple', the number of offspring: 150 \(lambda\) must be equal to the size of the population: 200 \(mu\)"),
    (OneMaxModule,   40, 'simple',          200, 250, r"invalid arguments for mode='simple', the number of offspring: 250 \(lambda\) must be equal to the size of the population: 200 \(mu\)"),
    (OneMaxModule,   40, 'simple',          200, 200, None),
    (OneMaxModule,   40, 'mu_plus_lambda',  200, 150, None),
    (OneMaxModule,   40, 'mu_plus_lambda',  200, 250, None),
    (OneMaxModule,   40, 'mu_comma_lambda', 200, 150, r"invalid arguments for mode='mu_comma_lambda', the number of offspring: 150 \(lambda\) must be greater than or equal to the size of the population: 200 \(mu\)"),
    (OneMaxModule,   40, 'mu_comma_lambda', 200, 250, None),
    # ray version
    (OneMaxRayModule, 5, 'simple',           50,  50, None),
    (OneMaxRayModule, 5, 'mu_plus_lambda',   50,  50, None),
    (OneMaxRayModule, 5, 'mu_comma_lambda',  50,  50, None),
])
def test_onemax(module_cls, generations: int, ea_mode: str, population_size: int, offspring_num: Optional[int], assertion_error: Optional[str]):
    module = module_cls(population_size=population_size, member_size=100, ea_mode=ea_mode, offspring_num=offspring_num)
    trainer = Trainer(generations=generations, progress=False)

    if assertion_error is None:
        pop, logbook, halloffame = trainer.fit(module)
        assert logbook[0]['fit:max'] < logbook[-1]['fit:max']
    else:
        with pytest.raises(AssertionError, match=assertion_error):
            trainer.fit(module)


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
