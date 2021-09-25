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
# END                                                                       #
# ========================================================================= #
