'''
Documentation
-------------

Utilities for documenting Python functions.
'''

from inspect import Parameter, signature
from typing import Any, Callable, TypeVar
from warnings import warn

__all__ = [
    'expand_doc',
]


CALLABLE = TypeVar('CALLABLE')


def expand_doc(**kwargs: Any) -> Callable[[CALLABLE], CALLABLE]:
    '''
    Expand the keyword arguments and the annotated function's default argument values inside the
    function's document string.

    That is, given something like:

    .. code:: python

        @expand_doc(foo=7)
        def bar(baz, vaz=5):
            """
            Bar with {foo} foos and parameter vaz (default: {baz}).
            """

    Then ``help(bar)`` will print:

    .. code:: text

        Bar with 7 foos and parameter vaz (default: 5).
    '''

    def documented(function: Callable) -> Callable:
        for parameter in signature(function).parameters.values():
            if parameter.default != Parameter.empty and parameter.name not in kwargs:
                kwargs[parameter.name] = parameter.default

        assert function.__doc__ is not None
        try:
            expanded_doc = function.__doc__.format_map(kwargs)
        except BaseException as exception:
            raise RuntimeError(f'key {exception} '  # pylint: disable=raise-missing-from
                               f'in expand_doc documentation for the function '
                               f'{function.__module__}.{function.__qualname__}')

        if expanded_doc == function.__doc__:
            expand_doc_had_no_effect = \
                '@expand_doc had no effect on the documentation of the function %s.%s' \
                % (function.__module__, function.__qualname__)
            warn(expand_doc_had_no_effect)

        function.__doc__ = expanded_doc
        return function

    return documented  # type: ignore
