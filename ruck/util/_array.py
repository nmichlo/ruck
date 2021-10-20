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


from typing import Sequence
from typing import Union

import numpy as np


# ========================================================================= #
# Array Util                                                                #
# ========================================================================= #


def arggroup(
    numbers: Union[Sequence, np.ndarray],
    axis=0,
    keep_order=True,
    return_unique: bool = False,
    return_index: bool = False,
    return_counts: bool = False,
):
    """
    Group all the elements of the array.
    - The returned groups contain the indices of
      the original position in the arrays.
    """

    # convert
    if not isinstance(numbers, np.ndarray):
        numbers = np.array(numbers)
    # checks
    if numbers.ndim == 0:
        raise ValueError('input array must have at least one dimension')
    if numbers.size == 0:
        return []
    # we need to obtain the sorted groups of
    unique, index, inverse, counts = np.unique(numbers, return_index=True, return_inverse=True, return_counts=True, axis=axis)
    # same as [ary[:idx[0]], ary[idx[0]:idx[1]], ..., ary[idx[-2]:idx[-1]], ary[idx[-1]:]]
    groups = np.split(ary=np.argsort(inverse, axis=0), indices_or_sections=np.cumsum(counts)[:-1], axis=0)
    # maintain original order
    if keep_order:
        add_order = index.argsort()  # the order that items were added in
        groups = [groups[i] for i in add_order]
    # return values
    results = [groups]
    if return_unique:  results.append(unique[add_order] if keep_order else unique)
    if return_index:   results.append(index[add_order]  if keep_order else index)
    if return_counts:  results.append(counts[add_order] if keep_order else counts)
    # unpack
    if len(results) == 1:
        return results[0]
    return results


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
