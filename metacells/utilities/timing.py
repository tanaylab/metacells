'''
Timing
------

The first step in achieving reasonable performance is identifying where most of the time is being
spent. The functions in this module allow to easily collect timing information about the relevant
functions or steps within functions in a controlled way, with low overhead, as opposed to collecting
information about all functions which has higher overheads and produces mountains of mostly
irrelevant data.
'''

import os
import sys
from contextlib import contextmanager
from functools import wraps
from threading import current_thread
from threading import local as thread_local
from time import perf_counter_ns, process_time_ns
from typing import (IO, Any, Callable, Dict, Iterator, List, NamedTuple,
                    Optional, TypeVar)

import metacells.utilities.documentation as utd
import metacells.utilities.logging as utl

__all__ = [
    'collect_timing',
    'flush_timing',
    'in_parallel_map',
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
TIMING_MODE = 'a'
TIMING_BUFFERING = 1
TIMING_FILE: Optional[IO] = None

LOG_ALL_STEPS = False

THREAD_LOCAL = thread_local()
COUNTED_THREADS = 0


@utd.expand_doc()
def collect_timing(
    collect: bool,
    path: str = TIMING_PATH,  # pylint: disable=used-prior-global-declaration
    mode: str = TIMING_MODE,  # pylint: disable=used-prior-global-declaration
    *,
    buffering: int = TIMING_BUFFERING  # pylint: disable=used-prior-global-declaration
) -> None:
    '''
    Specify whether, where and how to collect timing information.

    By default, we do not. Override this by setting the ``METACELLS_COLLECT_TIMING`` environment
    variable to ``true``, or by invoking this function from the main thread.

    By default, the data is written to the ``path`` is {path}, which is opened with the mode is
    {mode} and using the buffering is {buffering}. Override this by setting the
    ``METACELL_TIMING_PATH``, ``METACELL_TIMING_MODE`` and/or the ``METACELL_TIMING_BUFFERING``
    environment variables, or by invoking this function from the main thread.

    This will flush and close the previous timing file, if any.

    The file is written in CSV format (without headers). The first three fields are:

    * The invocation context (a ``.``-separated path of "relevant" function/step names).

    * The elapsed time (in nanoseconds) in this context (not counting nested contexts).

    * The CPU time (in nanoseconds) in this context (not counting nested contexts).

    This may be followed by a series of ``name,value`` pairs describing parameters of interest for
    this context, such as data sizes and layouts, to help understand the performance of the code.
    '''
    assert current_thread().name in ('#0', 'MainThread')

    global TIMING_PATH
    global TIMING_MODE
    global TIMING_BUFFERING
    global TIMING_FILE
    global COLLECT_TIMING

    if not path.endswith('.csv'):
        raise ValueError('The METACELL_TIMING_PATH: %s does not end with: .csv'
                         % path)

    TIMING_PATH = path
    TIMING_MODE = mode
    TIMING_BUFFERING = buffering

    if TIMING_FILE is not None:
        TIMING_FILE.flush()
        TIMING_FILE.close()
        TIMING_FILE = None

    if collect:
        TIMING_FILE = open(TIMING_PATH, TIMING_MODE,
                           buffering=TIMING_BUFFERING)

    COLLECT_TIMING = collect


def flush_timing() -> None:
    '''
    Flush the timing information, if we are collecting it.
    '''
    if TIMING_FILE is not None:
        TIMING_FILE.flush()


def in_parallel_map(map_index: int, process_index: int) -> None:
    '''
    Reconfigure timing collection when running in a parallel sub-process via
    :py:func:`metacells.utilities.parallel.parallel_map`.

    This will direct the timing information from ``<timing>.csv`` to
    ``<timing>.<map>.<process>.csv`` (where ``<timing>`` is from the original path, ``<map>`` is the
    serial number of the :py:func:`metacells.utilities.parallel.parallel_map` invocation, and
    ``<process>`` is the serial number of the process in the map).

    Collecting the timing of separate sub-processes to separate files allows us to freely write to
    them without locks and synchronizations which improves the performance (reduces the overhead of
    collecting timing information).

    You can just concatenate the files when the run is complete, or use a tool which automatically
    collects the data from all the files, such as :py:mod:`metacells.scripts.timing`.
    '''
    if COLLECT_TIMING:
        assert TIMING_PATH.endswith('.csv')
        collect_timing(True, '%s.%s.%s.csv'
                       % (TIMING_PATH[:-4], map_index, process_index))


def log_steps(log: bool) -> None:
    '''
    Whether to log every step invocation.

    By default, we do not. Override this by setting the ``METACELLS_LOG_ALL_STEPS`` environment
    variable to ``true`` or by invoking this function from the main thread.

    .. note::

        This only works if :py:func:`collect_timing` was set. It is a crude instrument to hunt for
        deadlocks, very-long-running numpy functions, and the like. Basically, if the program is
        taking 100% CPU and you have no idea what it is doing, turning this on and looking at the
        last logged step name would give you some idea of where it is stuck.
    '''
    global LOG_ALL_STEPS
    LOG_ALL_STEPS = log


if not 'sphinx' in sys.argv[0]:
    TIMING_PATH = os.environ.get('METACELL_TIMING_CSV', TIMING_PATH)
    TIMING_MODE = os.environ.get('METACELL_TIMING_MODE', TIMING_MODE)
    TIMING_BUFFERING = \
        int(os.environ.get('METACELL_TIMING_BUFFERING', str(TIMING_BUFFERING)))
    collect_timing({'true': True,
                    'false': False}[os.environ.get('METACELLS_COLLECT_TIMING',
                                                   str(COLLECT_TIMING)).lower()])
    log_steps({'true': True,
               'false': False}[os.environ.get('METACELLS_LOG_ALL_STEPS',
                                              str(LOG_ALL_STEPS)).lower()])


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


class GcStep(NamedTuple):
    '''
    Data about a GC collection step.
    '''

    #: The counters when we started the GC step.
    start: Counters

    #: The counters when we ended the GC step.
    stop: Counters


if COLLECT_TIMING:
    import gc

    GC_STEPS: List[GcStep] = []

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

        gc_step = GcStep(start=GC_START_POINT, stop=Counters.now())
        GC_STEPS.append(gc_step)
        GC_START_POINT = None

        gc_parameters = []
        for name, value in info.items():
            gc_parameters.append(name)
            gc_parameters.append(str(value))

        _print_timing('__gc__', gc_step.stop - gc_step.start, gc_parameters)

    gc.callbacks.append(_time_gc)


@contextmanager
def timed_step(name: str) -> Iterator[None]:  # pylint: disable=too-many-branches
    '''
    Collect timing information for a computation step.

    Expected usage is:

    .. code:: python

        with ut.timed_step("foo"):
            some_computation()

    If we are collecting timing information, then for every invocation, the program will append a
    line similar to:

    .. code:: text

        foo,elapsed_ns,123,cpu_ns,456

    To a timing log file (default: ``timing.csv``). Additional fields can be appended to the line
    using the ``metacells.utilities.timing.parameters`` function.

    If the ``name`` starts with a ``.`` of a ``_``, then it is prefixed with the names of the
    innermost surrounding step name (which must exist). This is commonly used to time sub-steps of a
    function.
    '''
    if not COLLECT_TIMING:
        yield None
        return

    steps_stack = getattr(THREAD_LOCAL, 'steps_stack', None)
    if steps_stack is None:
        steps_stack = THREAD_LOCAL.steps_stack = []

    parent_timing: Optional[StepTiming] = None
    if len(steps_stack) > 0:
        parent_timing = steps_stack[-1]
    if name[0] == '_':
        name = f'.{name[1:]}'
    if name[0] == '.':
        assert parent_timing is not None

    step_timing = StepTiming(name, parent_timing)
    steps_stack.append(step_timing)

    try:
        if LOG_ALL_STEPS:
            utl.logger().debug('{[( %s', step_timing.context)

        yield_point = Counters.now()
        yield None

    finally:
        back_point = Counters.now()
        total_times = back_point - yield_point

        global GC_STEPS
        gc_steps: List[GcStep] = []
        for gc_step in GC_STEPS:
            if yield_point.elapsed_ns <= gc_step.start.elapsed_ns \
                    and gc_step.stop.elapsed_ns <= back_point.elapsed_ns:
                total_times -= gc_step.stop - gc_step.start
            else:
                gc_steps.append(gc_step)
        GC_STEPS = gc_steps

        assert total_times.elapsed_ns >= 0
        assert total_times.cpu_ns >= 0
        if LOG_ALL_STEPS:
            utl.logger().debug('}]) %s', step_timing.context)

        steps_stack.pop()

        if parent_timing is not None:
            parent_timing.total_nested += total_times

        total_times -= step_timing.total_nested

        _print_timing(step_timing.context, total_times, step_timing.parameters)

        assert total_times.elapsed_ns >= 0
        assert total_times.cpu_ns >= 0


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
    example, ``timed_parameters(foo=2, bar=3)`` would add ``foo,2,bar,3`` to the line in
    ``timing.csv``.

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
    Automatically wrap each invocation of the decorated function with
    :py:func:`metacells.utilities.timing.timed_step` using the ``name`` (by default, the function's
    ``__qualname__``).

    Expected usage is:

    .. code:: python

        @ut.timed_call()
        def some_function(...):
            ...
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

        The context will be the empty string unless we are actually collecting timing.
    '''
    steps_stack = getattr(THREAD_LOCAL, 'steps_stack', None)
    if not steps_stack:
        return ''
    return steps_stack[-1].context


def current_step() -> Optional[StepTiming]:
    '''
    The timing collector for the innermost (current)
    :py:func:`metacells.utilities.timing.timed_step`, if any.
    '''
    if not COLLECT_TIMING:
        return None
    steps_stack = getattr(THREAD_LOCAL, 'steps_stack', None)
    if steps_stack is None or len(steps_stack) == 0:
        return None
    return steps_stack[-1]


# This is a circular dependency so having it at the end allows the exported symbols to be seen.
