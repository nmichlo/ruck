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


from typing import Literal
from typing import overload

import numpy as np
import numpy.typing as npt

# ========================================================================= #
# Array Util                                                                #
# ========================================================================= #


@overload
def arggroup(
    numbers: npt.ArrayLike,
    axis: int = 0,
    keep_order: bool = True,
) -> list[np.ndarray]: ...


@overload
def arggroup(
    numbers: npt.ArrayLike,
    axis: int = 0,
    keep_order: bool = True,
    *,
    return_unique: Literal[True],
    return_counts: Literal[True],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]: ...


def arggroup(
    numbers: npt.ArrayLike,
    axis: int = 0,
    keep_order: bool = True,
    return_unique: bool = False,
    return_counts: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Group all the elements of the array.
    - The returned groups contain the indices of
      the original position in the arrays.
    - `return_unique` and `return_counts` must be requested together.
    """
    assert return_unique == return_counts, "`return_unique` and `return_counts` must be requested together"
    # convert
    if not isinstance(numbers, np.ndarray):
        numbers = np.array(numbers)
    # checks
    if numbers.ndim == 0:
        raise ValueError("input array must have at least one dimension")
    if numbers.size == 0:
        return []
    # we need to obtain the sorted groups of
    unique, index, inverse, counts = np.unique(
        numbers, return_index=True, return_inverse=True, return_counts=True, axis=axis
    )
    # same as [ary[:idx[0]], ary[idx[0]:idx[1]], ..., ary[idx[-2]:idx[-1]], ary[idx[-1]:]]
    groups = np.split(ary=np.argsort(inverse, axis=0), indices_or_sections=np.cumsum(counts)[:-1], axis=0)
    # maintain original order
    if keep_order:
        add_order = index.argsort()  # the order that items were added in
        groups = [groups[i] for i in add_order]
    # return values
    if not return_unique:
        return groups
    unique_out = unique[add_order] if keep_order else unique
    counts_out = counts[add_order] if keep_order else counts
    return groups, unique_out, counts_out


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
