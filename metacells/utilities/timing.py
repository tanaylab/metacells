'''
Utilities for timing operations.
'''

import os
import sys
from contextlib import contextmanager
from threading import Lock, current_thread
from threading import local as thread_local
from time import perf_counter_ns, thread_time_ns
from typing import Any, Callable, Dict, Iterator, List, Optional, TextIO

import numpy as np  # type: ignore

__all__ = [
    'COLLECT_TIMING',
    'TIMING_PATH',
    'TIMING_BUFFERING',
    'step',
    'call',
    'parameters',
    'context',
    'current_step',
]

#: Whether to collect timing at all. Override this by setting the ``METACELLS_COLLECT_TIMING``
#: environment variable to ``true``.
COLLECT_TIMING = False

#: The path of the timing CSV file to write. Override this by setting the ``METACELL_TIMING_CSV``
#: environment variable to some path.
TIMING_PATH = 'timing.csv'

#: The buffering mode to use when writing to the timing CSV file. Override this by setting the
#: ``METACELL_TIMING_BUFFERING`` environment variable to ``0`` for no buffering, ``1`` for line
#: buffering, or the size of the buffer.
TIMING_BUFFERING = 1

if not 'sphinx' in sys.argv[0]:
    COLLECT_TIMING = \
        {'true': True,
         'false': False}[os.environ.get('METACELLS_COLLECT_TIMING',
                                        'False').lower()]
    TIMING_PATH = os.environ.get('METACELL_TIMING_CSV', 'timing.csv')
    TIMING_BUFFERING = int(os.environ.get('METACELL_TIMING_BUFFERING', '1'))

TIMING_FILE: Optional[TextIO] = None
THREAD_LOCAL = thread_local()
LOCK = Lock()

COUNTED_THREADS = 0


def _thread_index() -> int:
    index = getattr(THREAD_LOCAL, 'index', None)
    if index is None:
        with LOCK:
            global COUNTED_THREADS
            COUNTED_THREADS += 1
            index = THREAD_LOCAL.index = COUNTED_THREADS
    return index


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

    def __init__(self, name: str, parent: Optional['StepTiming']) -> None:
        '''
        Start collecting time for a named processing step.
        '''
        #: The parent step, if any.
        self.parent = parent

        if name[0] != '.':
            name = ';' + name

        if parent is None:
            assert not name[0] == '.'
            name = name[1:]

        #: The full context of the processing step.
        self.context: str = name if parent is None else parent.context + name

        #: Parameters of interest of the processing step.
        self.parameters: List[str] = []

        #: The thread the step was invoked in.
        self.thread_name = current_thread().name

        #: The amount of CPU used in nested steps in the same thread.
        self.total_nested = Counters()

        #: Amount of resources used in other thread by parallel code.
        self.other_thread = Counters()

        #: The set of other threads used.
        self.other_thread_mask = \
            np.zeros(1 + (os.cpu_count() or 1), dtype='bool')


if COLLECT_TIMING:
    import gc

    GC_START_POINT: Optional[Counters] = None

    def _time_gc(phase: str, info: Dict[str, Any]) -> None:
        steps_stack = getattr(THREAD_LOCAL, 'steps_stack', None)
        if not steps_stack:
            return

        global GC_START_POINT
        if phase == 'start':
            assert GC_START_POINT is None
            GC_START_POINT = Counters.now()
            return

        assert phase == 'stop'
        assert GC_START_POINT is not None

        gc_total_time = Counters.now() - GC_START_POINT
        GC_START_POINT = None

        parent_timing = steps_stack[-1]
        parent_timing.other_thread += gc_total_time

        gc_parameters = []
        for name, value in info.items():
            gc_parameters.append(name)
            gc_parameters.append(str(value))

        _print_timing('__gc__', gc_total_time, gc_parameters)

    gc.callbacks.append(_time_gc)


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

        foo,elapsed_ns,123,cpu_ns,456

    To a timing log file (default: ``timing.csv``). Additional fields can be appended to the line
    using the ``metacells.utilities.timing.parameters`` function.

    If the ``name`` starts with a ``.``, then it is prefixed with the names of the innermost
    surrounding step name (which must exist). This is commonly used to time sub-steps of a function.
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
        name = '_'

    else:
        if len(steps_stack) > 0:
            parent_timing = steps_stack[-1]
        assert name != '_'
        if name[0] == '.':
            assert parent_timing is not None

    start_point = Counters.now()
    step_timing = StepTiming(name, parent_timing)
    steps_stack.append(step_timing)

    try:
        # import sys
        # sys.stderr.write(step_timing.context + ' BEGIN {\n')
        # sys.stderr.flush()
        yield None
        # sys.stderr.write(step_timing.context + ' END }\n')
        # sys.stderr.flush()
    finally:
        total_times = Counters.now() - start_point
        assert total_times.elapsed_ns >= 0
        assert total_times.cpu_ns >= 0

        steps_stack.pop()

        if name == '_':
            assert parent_timing is not None
            total_times.elapsed_ns = 0
            parent_timing.other_thread += total_times
            parent_timing.other_thread_mask[_thread_index()] = True

        else:
            if parent_timing is not None:
                parent_timing.total_nested += total_times

            total_times -= step_timing.total_nested
            assert total_times.elapsed_ns >= 0
            assert total_times.cpu_ns >= 0
            total_times += step_timing.other_thread

            _print_timing(step_timing.context, total_times, step_timing.parameters,
                          step_timing.other_thread_mask.sum())


def _print_timing(
    invocation_context: str,
    total_times: Counters,
    step_parameters: Optional[List[str]] = None,
    other_threads_count: int = 0,
) -> None:
    with LOCK:
        global TIMING_FILE
        if TIMING_FILE is None:
            TIMING_FILE = \
                open(TIMING_PATH, 'a', buffering=TIMING_BUFFERING)
        text = [invocation_context,
                'elapsed_ns', str(total_times.elapsed_ns),
                'cpu_ns', str(total_times.cpu_ns)]
        if other_threads_count > 0:
            text.append('other_threads')
            text.append(str(other_threads_count))
        if step_parameters:
            text.extend(step_parameters)
        TIMING_FILE.write(','.join(text) + '\n')


def parameters(**kwargs: Any) -> None:
    '''
    Associate relevant timing parameters to the innermost ``timing.step``.

    The specified arguments are appended at the end of the generated ``timing.csv`` line. For
    example, ``parameters(foo=2, bar=3)`` would add ``foo,2,bar,3`` to the line ``timing.csv``.

    This allows tracking parameters which affect invocation time (such as array sizes), to help
    calibrate parameters such as ``minimal_invocations_per_batch`` for ``parallel_map`` and
    ``parallel_for``.
    '''
    step_timing = current_step()
    if step_timing is not None:
        for name, value in kwargs.items():
            step_timing.parameters.append(name)
            step_timing.parameters.append(str(value))


def call(name: Optional[str] = None) -> Callable[[Callable], Callable]:
    '''
    Automatically wrap each function invocation with ``timing.step`` using the ``name`` (by default,
    the function's qualified name).
    '''
    if not COLLECT_TIMING:
        return lambda function: function

    def wrap(function: Callable) -> Callable:
        def timed(*args: Any, **kwargs: Any) -> Any:
            with step(name or function.__qualname__):
                return function(*args, **kwargs)
        return timed

    return wrap


def context() -> str:
    '''
    Return the full current context (path of steps leading to the current point).

    .. note::

        The context will be empty unless we are collecting timing.

    .. note::

        This correctly tracks the context across threads when using ``parallel_for`` and
        ``parallel_map``.
    '''
    steps_stack = getattr(THREAD_LOCAL, 'steps_stack', None)
    if not steps_stack:
        return ''
    return steps_stack[-1].context


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
