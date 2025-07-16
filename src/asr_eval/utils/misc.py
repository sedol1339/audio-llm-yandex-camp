from __future__ import annotations

from typing import TypeVar
from itertools import groupby
from collections.abc import Iterable


T = TypeVar('T')


def groupby_into_spans(iterable: Iterable[T]) -> Iterable[tuple[T, int, int]]:
    '''
    Find spans of the same value in a sequence. Returns (value, start_index, end_index).
    
    list(groupby_enumerate(['x', 'x', 'b', 'a', 'a', 'a']))
    >>> [('x', 0, 2), ('b', 2, 3), ('a', 3, 6)]
    '''
    for key, group in groupby(enumerate(iterable), key=lambda x: x[1]):
        group = list(group)
        yield key, group[0][0], group[-1][0] + 1