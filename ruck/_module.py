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

from typing import Any
from typing import Dict
from typing import Generic
from typing import List
from typing import Sequence
from typing import TypeVar

import numpy as np

from ruck._history import StatsGroup
from ruck._member import Population
from ruck.util._args import HParamsMixin


# ========================================================================= #
# Module                                                                    #
# ========================================================================= #


T = TypeVar('T')


class EaModule(Generic[T], HParamsMixin):

    # OVERRIDABLE DEFAULTS

    def get_stats_groups(self) -> Dict[str, StatsGroup[T, Any]]:
        # default stats groups
        return {
            'fit': StatsGroup(
                lambda pop: [m.fitness for m in pop],
                min =lambda fitnesses: np.min(fitnesses,  axis=0).tolist(),
                max =lambda fitnesses: np.max(fitnesses,  axis=0).tolist(),
                mean=lambda fitnesses: np.mean(fitnesses, axis=0, dtype='float64').tolist(),
                std =lambda fitnesses: np.std(fitnesses,  axis=0, dtype='float64').tolist(),
            )
        }

    def get_progress_stats(self) -> Sequence[str]:
        # which stats are included in the progress bar
        # - values added by trainer
        return ('evals', 'fit:max')

    # REQUIRED

    def gen_starting_values(self) -> List[T]:
        raise NotImplementedError

    def generate_offspring(self, population: Population[T]) -> Population[T]:
        raise NotImplementedError

    def evaluate_values(self, values: List[T]) -> List[float]:
        raise NotImplementedError

    def select_population(self, population: Population[T], offspring: Population[T]) -> Population[T]:
        raise NotImplementedError


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
