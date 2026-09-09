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
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence

# ========================================================================= #
# iter                                                                      #
# ========================================================================= #


# NOTE:
#   Iterable: objects that return Iterators when passed to `iter()`
#   Iterator: return the next item when used with `next()`
#             every Iterator is ALSO an Iterable


def ipairs[T](items: Iterable[T]) -> Iterator[tuple[T, T]]:
    itr_a, itr_b = itertools.tee(items)
    itr_a = itertools.islice(itr_a, 0, None, 2)
    itr_b = itertools.islice(itr_b, 1, None, 2)
    return zip(itr_a, itr_b)


# ========================================================================= #
# lists                                                                     #
# ========================================================================= #


def chained[T](list_of_lists: Iterable[Iterable[T]]) -> list[T]:
    return list(itertools.chain(*list_of_lists))


def splits[T](items: Sequence[T], num_chunks: int, keep_empty: bool = False) -> list[list[T]]:
    """
    Divide `items` into `num_chunks` contiguous, roughly equal-sized chunks.
    - The first `len(items) % num_chunks` chunks get one extra item, matching
      the chunking behaviour of `np.array_split`.
    """
    # empty chunks are only produced if explicitly requested
    if not keep_empty:
        num_chunks = min(num_chunks, len(items))
    if num_chunks <= 0:
        raise ValueError("number of chunks must be greater than 0")
    # split into contiguous chunks, we return lists of lists, not a list
    # of tuples, so that it is compatible with ray.get
    base_size, remainder = divmod(len(items), num_chunks)
    chunks = []
    start = 0
    for i in range(num_chunks):
        size = base_size + (1 if i < remainder else 0)
        chunks.append(list(items[start : start + size]))
        start += size
    return chunks


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
