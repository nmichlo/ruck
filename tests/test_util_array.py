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

import numpy as np
import pytest

from ruck.util._array import arggroup


# ========================================================================= #
# TEST                                                                      #
# ========================================================================= #


def _assert_groups_equal(groups, targets):
    groups = [np.array(g).tolist() for g in groups]
    targets = [np.array(t).tolist() for t in targets]
    assert groups == targets


def test_arggroup_axis():
    numbers = [[2, 0], [2, 2], [0, 0], [2, 1], [2, 2], [2, 2], [0, 2], [1, 0], [1, 1], [1, 1]]
    targets            = [[2], [6], [7], [8, 9], [0], [3], [1, 4, 5]]
    targets_orig_order = [[0], [1, 4, 5], [2], [3], [6], [7], [8, 9]]
    # check that transposing everything works!
    _assert_groups_equal(arggroup(numbers,                      axis=0), targets_orig_order)
    _assert_groups_equal(arggroup(np.array(numbers),            axis=0), targets_orig_order)
    _assert_groups_equal(arggroup(np.array(numbers).T.tolist(), axis=1), targets_orig_order)
    _assert_groups_equal(arggroup(np.array(numbers).T,          axis=1), targets_orig_order)
    # check that transposing everything works!
    _assert_groups_equal(arggroup(numbers,                      axis=0, keep_order=False), targets)
    _assert_groups_equal(arggroup(np.array(numbers),            axis=0, keep_order=False), targets)
    _assert_groups_equal(arggroup(np.array(numbers).T.tolist(), axis=1, keep_order=False), targets)
    _assert_groups_equal(arggroup(np.array(numbers).T,          axis=1, keep_order=False), targets)


def test_arggroup():
    _assert_groups_equal(arggroup([]),               [])
    _assert_groups_equal(arggroup(np.zeros([0])),    [])
    _assert_groups_equal(arggroup(np.zeros([0, 0])), [])
    _assert_groups_equal(arggroup(np.zeros([1, 0])), [])
    _assert_groups_equal(arggroup(np.zeros([0, 1])), [])
    _assert_groups_equal(arggroup([1, 2, 3, 3]),    [[0], [1], [2, 3]])
    _assert_groups_equal(arggroup([3, 2, 1, 3]),    [[0, 3], [1], [2]])
    # check ndim=0
    with pytest.raises(ValueError, match=r'input array must have at least one dimension'): arggroup(0)
    with pytest.raises(ValueError, match=r'input array must have at least one dimension'): arggroup(np.zeros([]))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
