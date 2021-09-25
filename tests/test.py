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
import numpy as np
from ruck import Member
from ruck.functional import apply_mate


# ========================================================================= #
# TESTS                                                                     #
# ========================================================================= #


def test_mate_keep_order():
    random.seed(77)
    np.random.seed(77)
    # checks
    offspring = apply_mate(
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
    offspring = apply_mate(
        population=[Member(c) for c in 'abcde'],
        mate_fn=lambda a, b: (a.upper(), b.upper()),
        p=0.5,
        keep_order=False,
    )
    # done
    assert ''.join(m.value for m in offspring) == 'cdBAe'


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
