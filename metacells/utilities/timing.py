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


class Counters:
    '''
    The counters for the execution times.
    '''

    def __init__(self, *, elapsed_ns: int = 0, cpu_ns: int = 0) -> None:
        self.elapsed_ns = elapsed_ns  #: Elapsed time counter.
        self.cpu_ns = cpu_ns  #: CPU time counter.

    @staticmethod
    def now() -> 'Counters':
        '''
        Return the current value of the counters.
        '''
        return Counters(elapsed_ns=perf_counter_ns(), cpu_ns=thread_time_ns())

    def __iadd__(self, other: 'Counters') -> 'Counters':
        self.elapsed_ns += other.elapsed_ns
        self.cpu_ns += other.cpu_ns
        return self

    def __sub__(self, other: 'Counters') -> 'Counters':
        return Counters(elapsed_ns=self.elapsed_ns - other.elapsed_ns,
                        cpu_ns=self.cpu_ns - other.cpu_ns)

    def __isub__(self, other: 'Counters') -> 'Counters':
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
        #: The unique name of the processing step.
        self.step_name = name

        #: Parameters of interest of the processing step.
        self.parameters: List[Any] = []

        #: The thread the step was invoked in.
        self.thread_name = current_thread().name

        #: The amount of CPU used in nested steps in the same thread.
        self.total_nested = Counters()

        #: Amount of CPU used in other thread by parallel code.
        self.other_thread_cpu_ns = 0


@contextmanager
def step(name: str) -> Iterator[None]:  # pylint: disable=too-many-branches
    '''
    Collect timing information for a computation step.

    Expected usage is:

    .. code:: python

        with ut.step("foo"):
            some_computation()

    For every invocation, the program will append a line similar to:

    .. code:: text

        foo,123,456

    To a timing log file (``timing.csv`` by default). The first number is the CPU time used and the
    second is the elapsed time, both in nanoseconds. Time spent in other threads via
    ``parallel_map`` and/or ``parallel_for`` is automatically included. Time spent in nested timed
    step is not included, that is, the generated file contains just the "self" times of each step.

    If the ``name`` starts with a ``.``, then it is prefixed with the names of the innermost
    surrounding step name (which must exist). This is commonly used to time sub-steps of a function.
    For example, the following:

    .. code:: python

        @ut.call
        def foo(...):
            ...
            with ut.step(".bar"):
                ...
            with ut.step(".baz"):
                ...
            ...

    Will result in three lines written to the timing log file per invocation:

    .. code:: text

        foo,123,456      - time inside foo but outside the sub-steps
        foo.bar,123,456  - time inside the .bar sub-step of the foo function
        foo.baz,123,456  - time inside the .baz sub-step of the foo function

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
        if len(steps_stack) > 0:
            parent_timing = steps_stack[-1]
        assert name != ''
        if name[0] == '.':
            assert parent_timing is not None
            name = parent_timing.step_name + name

    start_point = Counters.now()
    step_timing = StepTiming(name)
    steps_stack.append(step_timing)

    try:
        yield None
    finally:
        total_times = Counters.now() - start_point
        assert total_times.elapsed_ns >= 0
        assert total_times.cpu_ns >= 0

        steps_stack.pop()

        if name == '':
            assert parent_timing is not None
            parent_timing.other_thread_cpu_ns += total_times.cpu_ns

        else:
            if parent_timing is not None:
                parent_timing.total_nested += total_times

            total_times -= step_timing.total_nested
            assert total_times.elapsed_ns >= 0
            assert total_times.cpu_ns >= 0
            total_times.cpu_ns += step_timing.other_thread_cpu_ns

            with LOCK:
                global TIMING_FILE
                if TIMING_FILE is None:
                    TIMING_FILE = \
                        open(TIMING_PATH, 'a', buffering=TIMING_BUFFERING)
                if len(step_timing.parameters) == 0:
                    TIMING_FILE.write('%s,%s,%s\n'
                                      % (name, total_times.cpu_ns, total_times.elapsed_ns))
                else:
                    TIMING_FILE.write('%s,%s,%s,%s\n'
                                      % (name, total_times.cpu_ns, total_times.elapsed_ns,
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


def call(name: Optional[str] = None) -> Callable[[Callable], Callable]:
    '''
    Automatically wrap each function invocation with ``timing.step`` using the ``name`` (by default,
    the function's qualified name).
    '''
    def wrap(function: Callable) -> Callable:
        def timed(*args: Any, **kwargs: Any) -> Any:
            with step(name or function.__qualname__):
                return function(*args, **kwargs)
        return timed

    return wrap
