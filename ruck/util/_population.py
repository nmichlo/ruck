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
import numpy as np
from ruck import Population


# ========================================================================= #
# Population Helper                                                         #
# ========================================================================= #


def population_fitnesses(population: Population, weights: Sequence[float] = None) -> np.ndarray:
    """
    Obtain an array of normalized fitness values from a population, the output
    shape always has two dimensions. (len(population), len(fitness))
    - Fitness values have ndim==0 or ndim==1 and are always expanded to ndim=1

    If weights are specified then we multiply by the normalized weights too.
    - Weights have ndim==0 or ndim==1 and are broadcast to match the fitness values.
    """
    fitnesses = np.array([m.fitness for m in population])
    # check dims
    if fitnesses.ndim == 1:
        fitnesses = fitnesses[:, None]
    assert fitnesses.ndim == 2
    # exit early
    if fitnesses.size == 0:
        return fitnesses
    # handle weights
    if weights is not None:
        weights = np.array(weights)
        # check dims
        if weights.ndim == 0:
            weights = weights[None]
        assert weights.ndim == 1
        # multiply
        fitnesses *= weights[None, :]
        assert fitnesses.ndim == 2
    # done
    return fitnesses  # shape: (len(population), len(fitness))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
