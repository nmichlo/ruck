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
from typing import List

from ruck._history import StatsGroup
from ruck._member import PopulationHint
from ruck._util.args import HParamsMixin


# ========================================================================= #
# Module                                                                    #
# ========================================================================= #


class EaModule(HParamsMixin):

    # OVERRIDABLE DEFAULTS

    def get_stats_groups(self) -> Dict[str, StatsGroup]:
        # additional stats to be recorded
        return {}

    def get_progress_stats(self):
        # which stats are included in the progress bar
        return ('evals', 'fit:max',)

    def evaluate_values(self, values: List[Any]) -> List[float]:
        # we include this here so we can easily override to add multi-threading support
        return [self.evaluate_value(value) for value in values]

    # REQUIRED

    def gen_starting_population(self) -> PopulationHint:
        raise NotImplementedError

    def generate_offspring(self, population: PopulationHint) -> PopulationHint:
        raise NotImplementedError

    def select_population(self, population: PopulationHint, offspring: PopulationHint) -> PopulationHint:
        raise NotImplementedError

    def evaluate_value(self, value: Any):
        raise NotImplementedError


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
