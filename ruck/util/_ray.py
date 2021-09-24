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

import functools
from typing import Any
from typing import List
from typing import Protocol
from typing import Sequence

import numpy as np
import ray
from ray.remote_function import RemoteFunction


# ========================================================================= #
# lists                                                                     #
# ========================================================================= #


def chained(list_of_lists: List[List[Any]]) -> List[Any]:
    return [item for items in list_of_lists for item in items]


def splits(items: List[Any], num_chunks: int = None, keep_empty: bool = False) -> List[List[Any]]:
    if num_chunks is None:
        num_chunks = _cpus()
    if not keep_empty:
        num_chunks = min(num_chunks, len(items))
    return [list(items) for items in np.array_split(items, num_chunks)]


# ========================================================================= #
# ray                                                                       #
# ========================================================================= #


class _RayFnHint(Protocol):
    def remote(self, *args, **kwargs) -> Any:
        pass
    def __call__(self, *args, **kwargs) -> Any:
        pass


@functools.lru_cache(maxsize=16)
def _to_remote_ray_fn(fn):
    if not isinstance(fn, RemoteFunction):
        fn = ray.remote(fn)
    return fn


@functools.lru_cache()
def _cpus():
    return ray.available_resources().get('CPU', 1)


def ray_map(ray_fn: _RayFnHint, items: Sequence[Any]) -> List[Any]:
    # make sure the function is a remote function
    ray_fn = _to_remote_ray_fn(ray_fn)
    # pass each item to ray and wait for the result
    return ray.get(list(map(ray_fn.remote, items)))


def ray_map_chunks(ray_fn: _RayFnHint, items: List[Any], num_chunks: int = None) -> List[Any]:
    # split items into chunks, and pass each chunk to function, then chain results back together
    return chained(ray_map(ray_fn, splits(items, num_chunks=num_chunks)))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
