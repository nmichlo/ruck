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


import itertools
import random
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Sequence
from typing import Tuple
from typing import TypeVar

import numpy as np


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


T = TypeVar('T')


# ========================================================================= #
# iter                                                                      #
# ========================================================================= #


# NOTE:
#   Iterable: objects that return Iterators when passed to `iter()`
#   Iterator: return the next item when used with `next()`
#             every Iterator is ALSO an Iterable


def ipairs(items: Iterable[T]) -> Iterator[Tuple[T, T]]:
    itr_a, itr_b = itertools.tee(items)
    itr_a = itertools.islice(itr_a, 0, None, 2)
    itr_b = itertools.islice(itr_b, 1, None, 2)
    return zip(itr_a, itr_b)


# ========================================================================= #
# lists                                                                     #
# ========================================================================= #


def chained(list_of_lists: Iterable[Iterable[T]]) -> List[T]:
    return list(itertools.chain(*list_of_lists))


def splits(items: Sequence[Any], num_chunks: int, keep_empty: bool = False) -> List[List[Any]]:
    # np.array_split will return empty elements if required
    if not keep_empty:
        num_chunks = min(num_chunks, len(items))
    # we return a lists of lists, not a list of
    # tuples so that it is compatible with ray.get
    return [list(items) for items in np.array_split(items, num_chunks)]


# ========================================================================= #
# random -- used for ruck.functional._algorithm                             #
# ========================================================================= #


def replaced_random_taken_pairs(fn: Callable[[T, T], Tuple[T, T]], items: Iterable[T], p: float, map_fn=map) -> List[T]:
    # shallow copy because we want to update elements in this list
    # - we need to take care to handle the special case where the length
    #   of items is odd, thus we cannot just call random_map with modified
    #   args using pairs and chaining the output
    items = list(items)
    # select random items
    idxs, vals = [], []
    for i, pair in enumerate(zip(items[0::2], items[1::2])):
        if random.random() < p:
            vals.append(pair)
            idxs.append(i)
    # map selected values
    vals = map_fn(lambda pair: fn(pair[0], pair[1]), vals)
    # update values
    for i, (v0, v1) in zip(idxs, vals):
        items[i*2+0] = v0
        items[i*2+1] = v1
    # done!
    return items


def replaced_random_taken_elems(fn: Callable[[T], T], items: Iterable[T], p: float, map_fn=map) -> List[T]:
    # shallow copy because we want to update elements in this list
    items = list(items)
    # select random items
    idxs, vals = [], []
    for i, v in enumerate(items):
        if random.random() < p:
            vals.append(v)
            idxs.append(i)
    # map selected values
    vals = map_fn(fn, vals)
    # update values
    for i, v in zip(idxs, vals):
        items[i] = v
    # done!
    return items


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
