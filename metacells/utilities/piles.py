'''
Piles
-----
'''

from math import ceil, floor
from typing import List

import numpy as np  # type: ignore

import metacells.utilities.timing as utm
import metacells.utilities.typing as utt

__all__ = [
    'random_piles',
    'group_piles',
]


@utm.timed_call()
def random_piles(
    elements_count: int,
    target_pile_size: int,
    *,
    random_seed: int = 0,
) -> utt.DenseVector:
    '''
    Split ``elements_count`` elements into piles of a size roughly equal to ``target_pile_size``.

    Return a vector specifying the pile index of each element.

    Specify a non-zero ``random_seed`` to make this replicable.
    '''
    assert target_pile_size > 0
    piles_count = elements_count / target_pile_size

    few_piles_count = floor(piles_count)
    many_piles_count = ceil(piles_count)

    if few_piles_count == many_piles_count:
        piles_count = few_piles_count

    else:
        few_piles_size = elements_count / few_piles_count
        many_piles_size = elements_count / many_piles_count

        few_piles_factor = few_piles_size / target_pile_size
        many_piles_factor = target_pile_size / many_piles_size

        assert few_piles_factor >= 1
        assert many_piles_factor >= 1

        if few_piles_factor < many_piles_factor:
            piles_count = few_piles_count
        else:
            piles_count = many_piles_count

    pile_of_elements_list: List[utt.DenseVector] = []

    minimal_pile_size = floor(elements_count / piles_count)
    extra_elements = elements_count - minimal_pile_size * piles_count
    assert 0 <= extra_elements < piles_count

    if extra_elements > 0:
        pile_of_elements_list.append(np.arange(extra_elements))
    for pile_index in range(piles_count):
        pile_of_elements_list.append(np.full(minimal_pile_size, pile_index))

    pile_of_elements = np.concatenate(pile_of_elements_list)
    assert pile_of_elements.size == elements_count

    np.random.seed(random_seed)
    return np.random.permutation(pile_of_elements)


@utm.timed_call()
def group_piles(
    group_of_elements: utt.DenseVector,
    group_of_groups: utt.DenseVector,
) -> utt.DenseVector:
    '''
    Group some elements into piles by grouping them, and then grouping the groups.

    Given the ``group_of_elements`` and for each such group, its larger ``group_of_groups``,
    compute the pile index of each element to be the group of the group it belongs to,
    and return a vector of the pile index of each element.

    .. note::

        Neither the ``group_of_elements`` nor the ``group_of_groups`` may contain outliers, that is,
        they must assign an valid group index to each element and group.
    '''
    group_of_group_of_elements = group_of_groups[group_of_elements]
    return group_of_group_of_elements
