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
from typing import List
from typing import Sequence
from typing import Tuple
from typing import TypeVar

import numpy as np


# ========================================================================= #
# lists                                                                     #
# ========================================================================= #


def chained(list_of_lists: List[List[Any]]) -> List[Any]:
    return [item for items in list_of_lists for item in items]


def splits(items: List[Any], num_chunks: int, keep_empty: bool = False) -> List[List[Any]]:
    if not keep_empty:
        num_chunks = min(num_chunks, len(items))
    return [list(items) for items in np.array_split(items, num_chunks)]


def replaced(targets: List[Any], idxs: Sequence[int], items: Sequence[int]):
    targets = list(targets)
    for i, v in zip(idxs, items):
        targets[i] = v
    return targets


def replaced_pairs(targets: List[Any], idx_item_pairs: Sequence[Tuple[int, Any]]):
    targets = list(targets)
    for i, v in idx_item_pairs:
        targets[i] = v
    return targets


def transposed(items, results: int) -> Tuple[List[Any], ...]:
    """
    Like `zip(*items)` but not an iterators
    and returns a tuple of lists instead
    """
    lists = [[] for i in range(results)]
    # get items
    for item in items:
        for l, v in zip(lists, item):
            l.append(v)
    # done
    return tuple(lists)


# ========================================================================= #
# random                                                                    #
# ========================================================================= #


T = TypeVar('T')


def random_map_pairs(fn: Callable[[T, T], Tuple[T, T]], items: Sequence[T], p: float, map_fn=map) -> List[T]:
    return chained(random_map(lambda v: fn(v[0], v[1]), ipairs(items), p, map_fn))


def random_map(fn: Callable[[T], T], items: Sequence[T], p: float, map_fn=map) -> List[T]:
    items = list(items)
    idxs, sel = transposed(itake_random(enumerate(items), p=p), results=2)
    sel = map_fn(fn, sel)
    return replaced(items, idxs, sel)


# ========================================================================= #
# iter                                                                      #
# ========================================================================= #


def itake_random(items, p: float):
    assert 0 <= p <= 1.0
    # exit early
    if p == 0:
        return
    # take items
    for item in items:
        if random.random() < p:
            yield item


def ipairs(items):
    itr_a, itr_b = itertools.tee(items)
    itr_a = itertools.islice(itr_a, 0, None, 2)
    itr_b = itertools.islice(itr_b, 1, None, 2)
    return zip(itr_a, itr_b)

    # equivalent slower alternative:
    # itr = iter(items)
    # while True:
    #     try:
    #         a = next(itr)
    #         b = next(itr)
    #     except StopIteration:
    #         return
    #     yield a, b


def imap_random(fn, items, p):
    for i, item in itake_random(enumerate(items), p=p):
        yield i, fn(item)


def imap_multi(*fns_last_is_items):
    """
    Example:
    >>> list(imap_multi(None, lambda x: x + 10, [[1, 2], [3, 4]]))
    >>> [(1, 12), (3, 14)]
    """
    *fns, items = fns_last_is_items
    for item in items:
        yield tuple((v if (fn is None) else fn(v)) for fn, v in zip(fns, item))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
