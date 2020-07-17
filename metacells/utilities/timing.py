'''
Utilities for timing operations.
'''

import os
from contextlib import contextmanager
from threading import Lock
from threading import local as thread_local
from time import perf_counter_ns, process_time_ns
from typing import Any, Callable, Iterator, Optional, TextIO

__all__ = [
    'step',
    'call',
]

COLLECT_TIMING = os.environ.get(
    'METACELLS_COLLECT_TIMING', 'False').lower() == 'true'

LOCK = Lock()

TIMING_FILE: Optional[TextIO] = None
TIMING_PATH: str = os.environ.get('METACELL_TIMING_FILE', 'timing.csv')
TIMING_BUFFERING: int = int(os.environ.get('METACELL_TIMING_BUFFERING', '1'))

THREAD_LOCAL = thread_local()


@contextmanager
def step(name: str) -> Iterator[None]:
    '''
    Collect timing information for a computation step.

    Expected usage is:

    .. code:: python

        with timing.step("foo"):
            some_computation()

    For every such invocation, the program will append a line similar to:

    ::

        123,123,foo

    To a timing log file (`timing.csv` by default). The first number is the total CPU time
    used and the second is the elapsed time, both in nanoseconds.
    '''
    if not COLLECT_TIMING:
        yield None
        return

    parameters_stack = getattr(THREAD_LOCAL, 'parameters_stack', None)
    if parameters_stack is None:
        parameters_stack = THREAD_LOCAL.parameters_stack = []
    parameters_stack.append(None)

    start_elapsed_ns = perf_counter_ns()
    start_process_ns = process_time_ns()
    try:
        yield None
    finally:
        stop_process_ns = process_time_ns()
        stop_elapsed_ns = perf_counter_ns()
        function_parameters = parameters_stack.pop()

    process_ns = stop_process_ns - start_process_ns
    elapsed_ns = stop_elapsed_ns - start_elapsed_ns

    with LOCK:
        global TIMING_FILE
        if TIMING_FILE is None:
            TIMING_FILE = open(TIMING_PATH, 'a', buffering=TIMING_BUFFERING)
        if function_parameters is None:
            TIMING_FILE.write('%s,%s,%s\n' % (process_ns, elapsed_ns, name))
        else:
            TIMING_FILE.write('%s,%s,%s,%s\n'
                              % (process_ns, elapsed_ns, name, function_parameters))


def parameters(*values: Any) -> None:
    '''
    Associate relevant timing parameters to the innermost ``timing.step``.

    The specified ``values`` are appended at the end of the generated ``timing.csv`` line. This
    allows tracking parameters which affect invocation time (such as array sizes), to help calibrate
    parameters such as ``minimal_invocations_per_batch`` for ``parallel_map`` and ``parallel_for``.
    '''
    if not COLLECT_TIMING:
        return
    parameters_stack = getattr(THREAD_LOCAL, 'parameters_stack', None)
    if parameters_stack is None or len(parameters_stack) == 0:
        return
    new_values = ','.join([str(value) for value in values])
    old_values = THREAD_LOCAL.parameters_stack[-1]
    if old_values is not None:
        new_values = old_values + ',' + new_values
    parameters_stack[-1] = new_values


def call(function: Callable) -> Callable[[Callable], Callable]:
    '''
    Automatically wrap each function invocation with ``timing.step`` using the function's name.
    '''
    if not COLLECT_TIMING:
        return function

    def timed(*args: Any, **kwargs: Any) -> Any:
        with step(function.__qualname__):
            return function(*args, **kwargs)
    return timed
