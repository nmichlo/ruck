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
from typing import Callable
from typing import List
from typing import Sequence

import ray
from ray import ObjectRef


# ========================================================================= #
# ray                                                                       #
# ========================================================================= #


def ray_map(remote_fn: Callable[[Any], ObjectRef], items: Sequence[Any]) -> List[Any]:
    """
    A simple ray alternative to `map`, input function should be
    a remote function that returns an object reference / future value
    """
    # pass each item to ray and wait for the result
    return ray.get(list(map(remote_fn, items)))


# ========================================================================= #
# ray - object store                                                        #
# ========================================================================= #


def ray_remote_put(fn = None, iter_results: bool = False, **ray_remote_kwargs):
    """
    Wrap a function using ray.remote but automatically put the
    results in the object store instead of returning the values!

    for example:
    >>> @ray.remote
    >>> def mate(a, b):
    >>>    a, b = R.mate_crossover_1d(a, b)
    >>>    return ray.put(a), ray.put(b)

    becomes:
    >>> @ray_remote_put(iter_results=True)
    >>> def mate(a, b):
    >>>     return R.mate_crossover_1d(a, b)

    or even:
    >>> mate = ray_remote_put(R.mate_crossover_1d, iter_results=True)
    """

    def wrapper(fn):
        # handle wrapping
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            result = fn(*args, **kwargs)
            # store values in the object store
            if iter_results:
                return tuple(ray.put(v) for v in result)
            else:
                return ray.put(result)
        # ray remote
        if ray_remote_kwargs:
            inner = ray.remote(**ray_remote_kwargs)(inner)
        else:
            inner = ray.remote(inner)
        # done!
        return inner

    # handle correct case
    if fn is None:
        return wrapper
    else:
        return wrapper(fn)


def ray_remote_puts(fn = None, **ray_remote_kwargs):
    """
    Like `ray_remote_put` but iterates over results.

    - This is the same as calling `ray_remote_put` with `iter_results=True`
    """
    return ray_remote_put(fn=fn, iter_results=True, **ray_remote_kwargs)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
