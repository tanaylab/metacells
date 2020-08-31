'''
Measure used elapsed and CPU times.
'''

import atexit
import os
import sys
from contextlib import contextmanager
from functools import wraps
from threading import current_thread
from threading import local as thread_local
from time import perf_counter_ns, process_time_ns
from typing import IO, Any, Callable, Dict, Iterator, List, Optional, TypeVar

__all__ = [
    'timing_file',
    'collect_timing',
    'log_steps',
    'timed_step',
    'timed_call',
    'timed_parameters',
    'context',
    'current_step',
    'StepTiming',
    'Counters',
]

COLLECT_TIMING = False

TIMING_PATH = 'timing.csv'
TIMING_MODE = 'timing.csv'
TIMING_BUFFERING = 1
TIMING_FILE: Optional[IO] = None

LOG_STEPS = False

THREAD_LOCAL = thread_local()
COUNTED_THREADS = 0


def timing_file(file: IO) -> None:
    '''
    Specify where to write the timing CSV lines.

    By default this is ``open('timing.csv', 'a', buffering=1)``. Override this by setting the
    ``METACELL_TIMING_PATH``, ``METACELL_TIMING_MODE`` and/or the ``METACELL_TIMING_BUFFERING``
    environment variables, or by invoking this function from the main thread.

    This will flush and close the previous timing file, if any.
    '''
    assert current_thread().name == 'MainThread'

    global TIMING_FILE
    if TIMING_FILE is not None:
        TIMING_FILE.flush()
        TIMING_FILE.close()
    TIMING_FILE = file


def collect_timing(collect: bool) -> None:
    '''
    Specify whether to collect timing information.

    By default, we do not. Override this by setting the ``METACELLS_COLLECT_TIMING`` environment
    variable to ``true``, or by invoking this function from the main thread.
    '''
    assert current_thread().name == 'MainThread'

    global COLLECT_TIMING

    if collect == COLLECT_TIMING:
        return

    if collect and TIMING_FILE is None:
        timing_file(open(TIMING_PATH, TIMING_MODE, buffering=TIMING_BUFFERING))

    COLLECT_TIMING = collect


def log_steps(log: bool) -> None:
    '''
    Whether to log every step invocation to ``sys.stderr``.

    By default, we do not. Override this by setting the ``METACELLS_LOG_STEPS`` environment variable
    to ``true`` or by invoking this function from the main thread.

    .. note::

        This only works if :py:func:`collect_timing` was set. It is a crude instrument to hunt for
        deadlocks, very-long-running numpy functions, and the like. Basically, if the program is
        taking 100% CPU and you have no idea what it is doing, turning this on would give you some
        idea of where it is stuck.
    '''
    global LOG_STEPS
    LOG_STEPS = log


if not 'sphinx' in sys.argv[0]:
    TIMING_PATH = os.environ.get('METACELL_TIMING_CSV', 'timing.csv')
    TIMING_MODE = os.environ.get('METACELL_TIMING_MODE', 'a')
    TIMING_BUFFERING = int(os.environ.get('METACELL_TIMING_BUFFERING', '1'))
    collect_timing({'true': True,
                    'false': False}[os.environ.get('METACELLS_COLLECT_TIMING',
                                                   'False').lower()])
    log_steps({'true': True,
               'false': False}[os.environ.get('METACELLS_LOG_STEPS',
                                              'False').lower()])


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
        return Counters(elapsed_ns=perf_counter_ns(), cpu_ns=process_time_ns())

    def __add__(self, other: 'Counters') -> 'Counters':
        return Counters(elapsed_ns=self.elapsed_ns + other.elapsed_ns,
                        cpu_ns=self.cpu_ns + other.cpu_ns)

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


if COLLECT_TIMING:
    OVERHEAD = Counters()

    def _print_overhead() -> None:
        _print_timing('__timing_overhead__', OVERHEAD)

    atexit.register(_print_overhead)


class StepTiming:  # pylint: disable=too-many-instance-attributes
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

        #: Amount of resources used in timing measurement functions
        self.overhead = Counters()


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
        parent_timing.total_nested += gc_total_time

        gc_parameters = []
        for name, value in info.items():
            gc_parameters.append(name)
            gc_parameters.append(str(value))

        _print_timing('__gc__', gc_total_time, gc_parameters)

    gc.callbacks.append(_time_gc)


@contextmanager
def timed_step(name: str) -> Iterator[None]:  # pylint: disable=too-many-branches,too-many-statements
    '''
    Collect timing information for a computation step.

    Expected usage is:

    .. code:: python

        with ut.timed_step("foo"):
            some_computation()

    For every invocation, the program will append a line similar to:

    .. code:: text

        foo,elapsed_ns,123,cpu_ns,456

    To a timing log file (default: ``timing.csv``). Additional fields can be appended to the line
    using the ``metacells.utilities.timing.parameters`` function.

    If the ``name`` starts with a ``.``, then it is prefixed with the names of the innermost
    surrounding step name (which must exist). This is commonly used to time sub-steps of a function.
    '''
    global OVERHEAD

    if not COLLECT_TIMING:
        yield None
        return

    start_point = Counters.now()

    steps_stack = getattr(THREAD_LOCAL, 'steps_stack', None)
    if steps_stack is None:
        steps_stack = THREAD_LOCAL.steps_stack = []

    parent_timing: Optional[StepTiming] = None
    if len(steps_stack) > 0:
        parent_timing = steps_stack[-1]
    if name[0] == '.':
        assert parent_timing is not None

    step_timing = StepTiming(name, parent_timing)
    steps_stack.append(step_timing)

    try:
        if LOG_STEPS:
            sys.stderr.write(step_timing.context + ' BEGIN {\n')
            sys.stderr.flush()
        yield_point = Counters.now()
        yield None
    finally:
        back_point = Counters.now()
        total_times = back_point - yield_point
        assert total_times.elapsed_ns >= 0
        assert total_times.cpu_ns >= 0
        if LOG_STEPS:
            sys.stderr.write(step_timing.context + ' END }\n')
            sys.stderr.flush()

        steps_stack.pop()

        if parent_timing is not None:
            parent_timing.total_nested += total_times

        total_times -= step_timing.total_nested
        assert total_times.elapsed_ns >= 0
        assert total_times.cpu_ns >= 0

        _print_timing(step_timing.context, total_times - step_timing.overhead,
                      step_timing.parameters)

        overhead = Counters.now() - back_point + yield_point - start_point
        if parent_timing is not None:
            overhead.elapsed_ns = 0
            parent_timing.overhead += overhead
        OVERHEAD += overhead


def _print_timing(
    invocation_context: str,
    total_times: Counters,
    step_parameters: Optional[List[str]] = None,
) -> None:
    gc_enabled = gc.isenabled()
    gc.disable()

    try:
        global TIMING_FILE
        if TIMING_FILE is None:
            TIMING_FILE = \
                open(TIMING_PATH, 'a', buffering=TIMING_BUFFERING)
        text = [invocation_context,
                'elapsed_ns', str(total_times.elapsed_ns),
                'cpu_ns', str(total_times.cpu_ns)]
        if step_parameters:
            text.extend(step_parameters)
        TIMING_FILE.write(','.join(text) + '\n')

    finally:
        if gc_enabled:
            gc.enable()


def timed_parameters(**kwargs: Any) -> None:
    '''
    Associate relevant timing parameters to the innermost
    :py:func:`metacells.utilities.timing.timed_step`.

    The specified arguments are appended at the end of the generated ``timing.csv`` line. For
    example, ``timed_parameters(foo=2, bar=3)`` would add ``foo,2,bar,3`` to the line
    in ``timing.csv``.

    This allows tracking parameters which affect invocation time (such as array sizes), to help
    identify the causes for the long-running operations.
    '''
    step_timing = current_step()
    if step_timing is not None:
        for name, value in kwargs.items():
            step_timing.parameters.append(name)
            step_timing.parameters.append(str(value))


CALLABLE = TypeVar('CALLABLE')


def timed_call(name: Optional[str] = None) -> Callable[[CALLABLE], CALLABLE]:
    '''
    Automatically wrap each function invocation with
    :py:func:`metacells.utilities.timing.timed_step` using the ``name`` (by default, the function's
    ``__qualname__``).
    '''
    if COLLECT_TIMING:
        def wrap(function: Callable) -> Callable:
            @wraps(function)
            def timed(*args: Any, **kwargs: Any) -> Any:
                with timed_step(name or function.__qualname__):
                    return function(*args, **kwargs)
            timed.__is_timed__ = True  # type: ignore
            return timed
    else:
        def wrap(function: Callable) -> Callable:
            function.__is_timed__ = True  # type: ignore
            return function

    return wrap  # type: ignore


def context() -> str:
    '''
    Return the full current context (path of :py:func:`metacells.utilities.timing.timed_step`-s
    leading to the current point).

    .. note::

        * The context will be empty unless we are collecting timing.
    '''
    steps_stack = getattr(THREAD_LOCAL, 'steps_stack', None)
    if not steps_stack:
        return ''
    return steps_stack[-1].context


def current_step() -> Optional[StepTiming]:
    '''
    The timing collector for the innermost (current)
    :py:func:`metacells.utilities.timing.timed_step`.
    '''
    if not COLLECT_TIMING:
        return None
    steps_stack = getattr(THREAD_LOCAL, 'steps_stack', None)
    if steps_stack is None or len(steps_stack) == 0:
        return None
    return steps_stack[-1]
