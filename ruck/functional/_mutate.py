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

from functools import wraps
from typing import Callable
from typing import TypeVar

import numpy as np


# ========================================================================= #
# Mutate Helper                                                             #
# ========================================================================= #


F = TypeVar('F')
T = TypeVar('T')
MutateFnHint = Callable[[T], T]


def check_mutation(fn: F) -> F:
    @wraps(fn)
    def wrapper(value: T, *args, **kwargs):
        mutated = fn(value, *args, **kwargs)
        assert mutated is not value, f'Mutate function: {fn} should return a new value'
        return mutated
    return wrapper


# ========================================================================= #
# Mutate                                                                    #
# ========================================================================= #


@check_mutation
def mutate_flip_bits(a: np.ndarray, p: float = 0.05) -> np.ndarray:
    return a ^ (np.random.random(a.shape) < p)


@check_mutation
def mutate_flip_bit_groups(a: np.ndarray, p: float = 0.05) -> np.ndarray:
    if np.random.random() < 0.5:
        # flip set bits
        return a ^ ((np.random.random(a.shape) < p) & a)
    else:
        # flip unset bits
        return a ^ ((np.random.random(a.shape) < p) & ~a)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
