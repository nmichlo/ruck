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
from typing import Tuple
from typing import TypeVar

import numpy as np


# ========================================================================= #
# Mate Helper                                                               #
# ========================================================================= #


F = TypeVar('F')
T = TypeVar('T')
MateFnHint = Callable[[T, T], Tuple[T, T]]


def check_mating(fn: F) -> F:
    @wraps(fn)
    def wrapper(value_a: T, value_b: T, *args, **kwargs) -> Tuple[T, T]:
        mated_a, mated_b = fn(value_a, value_b, *args, **kwargs)
        assert mated_a is not value_a, f'Mate function: {fn} should return new values'
        assert mated_a is not value_b, f'Mate function: {fn} should return new values'
        assert mated_b is not value_a, f'Mate function: {fn} should return new values'
        assert mated_b is not value_b, f'Mate function: {fn} should return new values'
        return mated_a, mated_b
    return wrapper


# ========================================================================= #
# Mate                                                                      #
# ========================================================================= #


@check_mating
def mate_crossover_1d(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert a.ndim == 1
    assert a.shape == b.shape
    # get slice
    i, j = np.random.randint(0, len(a), size=2)
    i, j = min(i, j), max(i, j)
    # generate new arrays
    new_a = np.concatenate([a[:i], b[i:j], a[j:]], axis=0)
    new_b = np.concatenate([b[:i], a[i:j], b[j:]], axis=0)
    return new_a, new_b


@check_mating
def mate_crossover_nd(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert a.ndim >= 1
    assert a.shape == b.shape
    # get hypercube
    I, J = np.random.randint(0, a.shape), np.random.randint(0, b.shape)
    I, J = np.minimum(I, J), np.maximum(I, J)
    # generate slices
    slices = tuple(slice(i, j, None) for i, j in zip(I, J))
    # copy arrays and set values
    new_a = np.copy(a)
    new_b = np.copy(b)
    new_a[slices] = b[slices]
    new_b[slices] = a[slices]
    return new_a, new_b


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
