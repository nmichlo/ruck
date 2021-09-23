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


# ========================================================================= #
# From https://github.com/nmichlo/disent
# ========================================================================= #


def random_choice_prng(a, size=None, replace=True, p=None):
    # create seeded pseudo random number generator
    # - built in np.random.choice cannot handle large values: https://github.com/numpy/numpy/issues/5299#issuecomment-497915672
    # - PCG64 is the default: https://numpy.org/doc/stable/reference/random/bit_generators/index.html
    # - PCG64 has good statistical properties and is fast: https://numpy.org/doc/stable/reference/random/performance.html
    g = np.random.Generator(np.random.PCG64(seed=np.random.randint(0, 2**32)))
    # sample indices
    choices = g.choice(a, size=size, replace=replace, p=p)
    # done!
    return choices


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
