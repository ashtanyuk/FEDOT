from copy import deepcopy
from typing import (Any, List)
from functools import partial

from log_calls import record_history

from fedot.core.composer.optimisers.selection import SelectionTypesEnum, individuals_selection
from fedot.core.utils import ComparableEnum as Enum


class GeneticSchemeTypesEnum(Enum):
    steady_state = 'steady_state'
    generational = 'generational'
    parameter_free = 'parameter_free'


@record_history(enabled=False)
def inheritance(type: GeneticSchemeTypesEnum, selection_types: List[SelectionTypesEnum],
                prev_population: List[Any], new_population: List[Any], max_size: int) -> List[Any]:
    steady_state_scheme = partial(steady_state_inheritance, selection_types, prev_population, new_population, max_size)
    generational_scheme = partial(direct_inheritance, new_population, max_size)
    inheritance_type_by_genetic_scheme = {
        GeneticSchemeTypesEnum.generational: generational_scheme,
        GeneticSchemeTypesEnum.steady_state: steady_state_scheme,
        GeneticSchemeTypesEnum.parameter_free: steady_state_scheme
    }
    return inheritance_type_by_genetic_scheme[type]()


@record_history(enabled=False)
def steady_state_inheritance(selection_types: List[SelectionTypesEnum],
                             prev_population: List[Any],
                             new_population: List[Any], max_size: int):
    return individuals_selection(types=selection_types, individuals=prev_population + new_population, pop_size=max_size)


@record_history(enabled=False)
def direct_inheritance(new_population: List[Any], max_size: int):
    return deepcopy(new_population[:max_size])
