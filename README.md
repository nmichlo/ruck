
<p align="center">
    <h1 align="center">🧬 Ruck 🏉</h1>
    <p align="center">
        <i>Performant evolutionary algorithms for Python</i>
    </p>
</p>

<p align="center">
    <a href="https://choosealicense.com/licenses/mit/">
        <img alt="license" src="https://img.shields.io/github/license/nmichlo/ruck?style=flat-square&color=lightgrey"/>
    </a>
    <a href="https://pypi.org/project/ruck">
        <img alt="python versions" src="https://img.shields.io/pypi/pyversions/ruck?style=flat-square"/>
    </a>
    <a href="https://pypi.org/project/ruck">
        <img alt="pypi version" src="https://img.shields.io/pypi/v/ruck?style=flat-square&color=blue"/>
    </a>
    <a href="https://github.com/nmichlo/ruck/actions?query=workflow%3Atest">
        <img alt="tests status" src="https://img.shields.io/github/workflow/status/nmichlo/ruck/test?label=tests&style=flat-square"/>
    </a>
</p>

<p align="center">
    <p align="center">
        Visit the <a href="https://github.com/nmichlo/ruck/tree/main/examples/">examples</a> to get started, or browse the  <a href="https://github.com/nmichlo/ruck/releases">releases</a>.
    </p>
    <p align="center">
        <a href="https://github.com/nmichlo/ruck/issues/new/choose">Contributions</a> are welcome!
    </p>
</p>

------------------------

## Goals

Ruck aims to fill the following criteria:

1. Provide **high quality**, **readable** implementations of algorithms.
2. Be easily **extensible** and **debuggable**.
3. Performant while maintaining its simplicity.

## Features

Ruck has various features that will be expanded upon in time
- 📦 Modular evolutionary systems inspired by pytorch lightning
  + Helps organise code & arguably looks clean
  
- ➕ Multi-Objective optimisation support
  + Optionally optimised version of NSGA-II if `numba` is installed, over 65x faster
  
- 🏎 Optional multithreading support with `ray`, including helper functions

- 🏭 Factory methods for simple evolutionary algorithms

- 🧪 Various helper functions for selection, mutation and mating


## Citing Ruck

Please use the following citation if you use Ruck in your research:

```bibtex
@Misc{Michlo2021Ruck,
  author =       {Nathan Juraj Michlo},
  title =        {Ruck - Performant evolutionary algorithms for Python},
  howpublished = {Github},
  year =         {2021},
  url =          {https://github.com/nmichlo/ruck}
}
```

## Overview

Ruck takes inspiration from PyTorch Lightning's module system. The population creation,
offspring, evaluation and selection steps are all contained within a single module inheriting
from `EaModule`. While the training logic and components are separated into its own class.

`Members` of a `Population` (A list of Members) are intended to be read-only. Modifications should not
be made to members, instead new members should be created with the modified values. This enables us to
easily implement efficient multi-threading, see below!

The trainer automatically constructs `HallOfFame` and `LogBook` objects which keep track of your
population and offspring. `EaModule` provides defaults for `get_stats_groups` and `get_progress_stats`
that can be overridden if you wish to customize the tracked statistics and statistics displayed by tqdm.


### Minimal OneMax Example

```python
import random
import numpy as np
from ruck import *


class OneMaxMinimalModule(EaModule):
    """
    Minimal onemax example
    - The goal is to flip all the bits of a boolean array to True
    - Offspring are generated as bit flipped versions of the previous population
    - Selection tournament is performed between the previous population and the offspring
    """

    # evaluate unevaluated members according to their values
    def evaluate_values(self, values):
        return [v.sum() for v in values]

    # generate 300 random members of size 100 with 50% bits flipped
    def gen_starting_values(self):
        return [np.random.random(100) < 0.5 for _ in range(300)]

    # randomly flip 5% of the bits of each each member in the population
    # the previous population members should never be modified
    def generate_offspring(self, population):
        return [Member(m.value ^ (np.random.random(m.value.shape) < 0.05)) for m in population]

    # selection tournament between population and offspring
    def select_population(self, population, offspring):
        combined = population + offspring
        return [max(random.sample(combined, k=3), key=lambda m: m.fitness) for _ in range(len(population))]


if __name__ == '__main__':
    # create and train the population
    module = OneMaxMinimalModule()
    pop, logbook, halloffame = Trainer(generations=100, progress=True).fit(module)

    print('initial stats:', logbook[0])
    print('final stats:', logbook[-1])
    print('best member:', halloffame.members[0])
```

### Advanced OneMax Example

Ruck provides various helper functions and implementations of evolutionary algorithms for convenience.
The following example makes use of these additional features so that components and behaviour can
easily be swapped out.

The three basic evolutionary algorithms provided are based around [deap's](http://www.github.com/deap/deap)
default algorithms from `deap.algorithms`. These basic evolutionary algorithms can be created from
`ruck.functional.make_ea`. We provide the alias `ruck.R` for `ruck.functional`. `R.make_ea` supports
the following modes: `simple`, `mu_plus_lambda` and `mu_comma_lambda`.


<details><summary><b>Code Example</b></summary>
<p>

```python
"""
OneMax serial example based on:
https://github.com/DEAP/deap/blob/master/examples/ga/onemax_numpy.py
"""

import functools
import numpy as np
from ruck import *


class OneMaxModule(EaModule):

    def __init__(
        self,
        population_size: int = 300,
        offspring_num: int = None,  # offspring_num (lambda) is automatically set to population_size (mu) when `None`
        member_size: int = 100,
        p_mate: float = 0.5,
        p_mutate: float = 0.5,
        ea_mode: str = 'simple'
    ):
        # save the arguments to the .hparams property. values are taken from the
        # local scope so modifications can be captured if the call to this is delayed.
        self.save_hyperparameters()
        # implement the required functions for `EaModule`
        self.generate_offspring, self.select_population = R.make_ea(
            mode=self.hparams.ea_mode,
            offspring_num=self.hparams.offspring_num,
            mate_fn=R.mate_crossover_1d,
            mutate_fn=functools.partial(R.mutate_flip_bit_groups, p=0.05),
            select_fn=functools.partial(R.select_tournament, k=3),
            p_mate=self.hparams.p_mate,
            p_mutate=self.hparams.p_mutate,
        )

    def evaluate_values(self, values):
        return map(np.sum, values)

    def gen_starting_values(self) -> Population:
        return [
            np.random.random(self.hparams.member_size) < 0.5
            for i in range(self.hparams.population_size)
        ]


if __name__ == '__main__':
    # create and train the population
    module = OneMaxModule(population_size=300, member_size=100)
    pop, logbook, halloffame = Trainer(generations=40, progress=True).fit(module)

    print('initial stats:', logbook[0])
    print('final stats:', logbook[-1])
    print('best member:', halloffame.members[0])
```

</p>
</details>

### Multithreading OneMax Example (Ray)

If we need to scale up the computational requirements, for example requiring increased
member and population sizes, the above serial implementations will soon run into performance problems.

The basic Ruck implementations of various evolutionary algorithms are designed around a `map`
function that can be swapped out to add multi-threading support. We can easily do this using
[ray](https://github.com/ray-project/ray) and we even provide various helper functions that
enhance ray support.

1. We begin by placing member's values into shared memory using ray's read-only object store
and the `ray.put` function. These [ObjectRef's](https://docs.ray.io/en/latest/memory-management.html)
values point to the original `np.ndarray` values. When retrieved with `ray.get` they obtain the original
arrays using an efficient zero-copy procedure. This is advantageous over something like Python's multiprocessing module which uses
expensive pickle operations to pass data around.

2. The second step is to swap out the aforementioned `map` function in the previous example to a
multiprocessing equivalent. We use `ray.remote` along with `ray.get`, and provide the `ray_map` function
that has the same API as python map, but accepts `ray.remote(my_fn).remote` values instead.

3. Finally we need to update our `mate` and `mutate` functions to handle `ObjectRef`s, we provide a convenient
wrapper to automatically call `ray.put` on function results so that you can re-use your existing code.

<details><summary><b>Code Example</b></summary>
<p>

```python
"""
OneMax parallel example using ray's object store.

8 bytes * 1_000_000 * 128 members ~= 128 MB of memory to store this population.
This is quite a bit of processing that needs to happen! But using ray
and its object store we can do this efficiently!
"""

from functools import partial
import numpy as np
from ruck import *
from ruck.external.ray import *


class OneMaxRayModule(EaModule):

    def __init__(
        self,
        population_size: int = 300,
        offspring_num: int = None,  # offspring_num (lambda) is automatically set to population_size (mu) when `None`
        member_size: int = 100,
        p_mate: float = 0.5,
        p_mutate: float = 0.5,
        ea_mode: str = 'mu_plus_lambda'
    ):
        self.save_hyperparameters()
        # implement the required functions for `EaModule`
        self.generate_offspring, self.select_population = R.make_ea(
            mode=self.hparams.ea_mode,
            offspring_num=self.hparams.offspring_num,
            # decorate the functions with `ray_remote_put` which automatically
            # `ray.get` arguments that are `ObjectRef`s and `ray.put`s returned results
            mate_fn=ray_remote_puts(R.mate_crossover_1d).remote,
            mutate_fn=ray_remote_put(R.mutate_flip_bit_groups).remote,
            # efficient to compute locally
            select_fn=partial(R.select_tournament, k=3),
            p_mate=self.hparams.p_mate,
            p_mutate=self.hparams.p_mutate,
            # ENABLE multiprocessing
            map_fn=ray_map,
        )
        # eval function, we need to cache it on the class to prevent
        # multiple calls to ray.remote. We use ray.remote instead of
        # ray_remote_put like above because we want the returned values
        # not object refs to those values.
        self._ray_eval = ray.remote(np.mean).remote

    def evaluate_values(self, values):
        # values is a list of `ray.ObjectRef`s not `np.ndarray`s
        # ray_map automatically converts np.sum to a `ray.remote` function which
        # automatically handles `ray.get`ing of `ray.ObjectRef`s passed as arguments
        return ray_map(self._ray_eval, values)

    def gen_starting_values(self):
        # generate objects and place in ray's object store
        return [
            ray.put(np.random.random(self.hparams.member_size) < 0.5)
            for i in range(self.hparams.population_size)
        ]


if __name__ == '__main__':
    # initialize ray to use the specified system resources
    ray.init()

    # create and train the population
    module = OneMaxRayModule(population_size=128, member_size=1_000_000)
    pop, logbook, halloffame = Trainer(generations=200, progress=True).fit(module)

    print('initial stats:', logbook[0])
    print('final stats:', logbook[-1])
    print('best member:', halloffame.members[0])
```

</p>
</details>
