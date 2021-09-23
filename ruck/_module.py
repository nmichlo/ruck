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

from ruck._history import StatsGroup
from ruck._member import PopulationHint
from ruck._util.args import HParamsMixin


# ========================================================================= #
# Module                                                                    #
# ========================================================================= #


class EaModule(HParamsMixin):

    # OVERRIDE

    def get_stats_groups(self) -> Dict[str, StatsGroup]:
        return {}

    def get_progress_stats(self):
        return ('evals', 'fit:max',)

    @property
    def num_generations(self) -> int:
        raise NotImplementedError

    def gen_starting_population(self) -> PopulationHint:
        raise NotImplementedError

    def generate_offspring(self, population: PopulationHint) -> PopulationHint:
        raise NotImplementedError

    def select_population(self, population: PopulationHint, offspring: PopulationHint) -> PopulationHint:
        raise NotImplementedError

    def evaluate_member(self, value: Any) -> float:
        raise NotImplementedError


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
