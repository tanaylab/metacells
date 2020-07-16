'''
Utilities for documentation generation.
'''

from inspect import Parameter, signature
from typing import Any, Callable

__all__ = [
    'expand_doc',
]


def expand_doc(**kwargs: Any) -> Callable[[Callable], Callable]:
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
            if parameter.default != Parameter.empty:
                kwargs[parameter.name] = parameter.default
        assert function.__doc__ is not None
        function.__doc__ = function.__doc__.format_map(kwargs)
        return function

    return documented
