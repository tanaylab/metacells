'''
Utilities for timing operations.
'''

import os
from contextlib import contextmanager
from threading import Lock, current_thread
from threading import local as thread_local
from time import perf_counter_ns, thread_time_ns
from typing import Any, Callable, Iterator, List, Optional, TextIO

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


class Snapshot:
    '''
    A snapshot of the state of CPU usage.
    '''

    def __init__(self, *, elapsed_ns: int = 0, cpu_ns: int = 0) -> None:
        self.elapsed_ns = elapsed_ns  #: Elapsed time counter.
        self.cpu_ns = cpu_ns  #: CPU time counter.

    @staticmethod
    def now() -> 'Snapshot':
        '''
        Return the current snapshot of the state of the CPU usage.
        '''
        return Snapshot(elapsed_ns=perf_counter_ns(), cpu_ns=thread_time_ns())

    def __iadd__(self, other: 'Snapshot') -> 'Snapshot':
        self.elapsed_ns += other.elapsed_ns
        self.cpu_ns += other.cpu_ns
        return self

    def __isub__(self, other: 'Snapshot') -> 'Snapshot':
        self.elapsed_ns -= other.elapsed_ns
        self.cpu_ns -= other.cpu_ns
        return self


class StepTiming:
    '''
    Timing information for some named processing step.
    '''

    def __init__(self, name: str) -> None:
        '''
        Start collecting time for a named processing step.
        '''
        self.step_name = name  #: The unique name of the processing step.
        #: Parameters of interest of the processing step.
        self.parameters: List[Any] = []
        #: The thread the step was invoked in.
        self.thread_name = current_thread().name
        self.start_point = Snapshot.now()  #: The point when the step started.


@contextmanager
def step(name: str) -> Iterator[None]:  # pylint: disable=too-many-branches
    '''
    Collect timing information for a computation step.

    Expected usage is:

    .. code:: python

        with timing.step("foo"):
            some_computation()

    For every such invocation, the program will append a line similar to:

    ::

        123,123,foo

    To a timing log file (`timing.csv` by default). The first number is the CPU time used and the
    second is the elapsed time, both in nanoseconds. Time spent in other threads via
    ``parallel_map`` and/or ``parallel_for`` is automatically included. Time spent in nested timed
    step is not included, that is, the generated file contains just the "self" times of each step.
    '''
    if not COLLECT_TIMING:
        yield None
        return

    steps_stack = getattr(THREAD_LOCAL, 'steps_stack', None)
    if steps_stack is None:
        steps_stack = THREAD_LOCAL.steps_stack = []

    parent_timing: Optional[StepTiming] = None
    if isinstance(name, StepTiming):
        parent_timing = name
        if parent_timing.thread_name == current_thread().name:
            yield None
            return
        name = ''

    else:
        assert name != ''
        if name[0] == '.':
            assert len(steps_stack) > 0
            name = steps_stack[-1].step_name + name

    step_timing = StepTiming(name)
    steps_stack.append(step_timing)

    try:
        yield None
    finally:
        total_times = Snapshot.now()
        total_times -= step_timing.start_point
        assert total_times.elapsed_ns >= 0
        assert total_times.cpu_ns >= 0

    steps_stack.pop()

    if name == '':
        assert parent_timing is not None
        parent_timing.start_point.cpu_ns -= total_times.cpu_ns
        return

    for parent_step_timing in steps_stack:
        parent_step_timing.start_point += total_times

    with LOCK:
        global TIMING_FILE
        if TIMING_FILE is None:
            TIMING_FILE = open(TIMING_PATH, 'a', buffering=TIMING_BUFFERING)
        if len(step_timing.parameters) == 0:
            TIMING_FILE.write('%s,%s,%s\n'
                              % (total_times.cpu_ns, total_times.elapsed_ns, name))
        else:
            TIMING_FILE.write('%s,%s,%s,%s\n'
                              % (total_times.cpu_ns, total_times.elapsed_ns, name,
                                 ','.join([str(parameter)
                                           for parameter
                                           in step_timing.parameters])))


def current_step() -> Optional[StepTiming]:
    '''
    The timing collector for the innermost (current) step.
    '''
    if not COLLECT_TIMING:
        return None
    steps_stack = getattr(THREAD_LOCAL, 'steps_stack', None)
    if steps_stack is None or len(steps_stack) == 0:
        return None
    return steps_stack[-1]


def parameters(*values: Any) -> None:
    '''
    Associate relevant timing parameters to the innermost ``timing.step``.

    The specified ``values`` are appended at the end of the generated ``timing.csv`` line. This
    allows tracking parameters which affect invocation time (such as array sizes), to help calibrate
    parameters such as ``minimal_invocations_per_batch`` for ``parallel_map`` and ``parallel_for``.
    '''
    step_timing = current_step()
    if step_timing is not None:
        step_timing.parameters.extend(values)


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
