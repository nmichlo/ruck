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
from typing import Sequence

import ray
from ray.remote_function import RemoteFunction


# ========================================================================= #
# ray                                                                       #
# ========================================================================= #


@functools.lru_cache(maxsize=16)
def _to_remote_ray_fn(fn):
    if not isinstance(fn, RemoteFunction):
        fn = ray.remote(fn)
    return fn


def ray_map(ray_fn, items: Sequence[Any]) -> List[Any]:
    """
    A more convenient alternative to `ray.util.multiprocessing.Pool`s `map` function!
    Using a similar API to python `map`, except returning a list of mapped values
    instead of an iterable.

    The advantage of this functions it that we automatically wrap passed functions to
    ray.remote functions, also enabling automatic getting of ObjectRef values.
    """
    # make sure the function is a remote function
    ray_fn = _to_remote_ray_fn(ray_fn)
    # pass each item to ray and wait for the result
    return ray.get(list(map(ray_fn.remote, items)))


# ========================================================================= #
# ray - object store                                                        #
# ========================================================================= #


def ray_refs_wrapper(fn = None, get: bool = True, put: bool = True, iter_results: bool = False):
    """
    Wrap a function so that we automatically ray.get
    all the arguments and ray.put the result.

    iter_results=True instead treats the result as an
    iterable and applies ray.put to each result item

    for example:
    >>> def mate(a, b):
    >>>    a, b = ray.get(a), ray.get(b)
    >>>    a, b = R.mate_crossover_1d(a, b)
    >>>    return ray.put(a), ray.put(b)

    becomes:
    >>> @ray_refs_wrapper(iter_results=True)
    >>> def mate(a, b):
    >>>     return R.mate_crossover_1d(a, b)
    """

    def wrapper(fn):
        @functools.wraps(fn)
        def inner(*args):
            # get values from object store
            if get:
                args = (ray.get(v) for v in args)
            # call function
            result = fn(*args)
            # store values in the object store
            if put:
                if iter_results:
                    result = tuple(ray.put(v) for v in result)
                else:
                    result = ray.put(result)
            # done!
            return result
        return inner

    # handle correct case
    if fn is None:
        return wrapper
    else:
        return wrapper(fn)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
