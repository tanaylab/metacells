'''
Parallel
--------

Due to the notorious GIL, using multiple Python threads is useless. This leaves us with two options
for using multiple CPUs:

* Use multiple threads in the internal C++ implementation of some Python functions;
  this is done by both numpy and the C++ extension functions provided here, and works even for
  reasonably small sized work, such as sorting each of the rows of a large matrix.

* Use Python multi-processing. This is costly and works only for large sized work, such as
  computing metacells for different piles.

Each of these two approaches works tolerably well on its own, even though both are sub-optimal. The
problem starts when we want to combine them. Consider a server with 50 processors. Invoking
``corrcoef`` on a large matrix will use them all. This is great if one computes metacells for a
single pile. Suppose, however, you want to compute metacells for 50 piles, and do so using
multi-processing. Each and every of the 50 sub-processes will invoke ``corcoeff`` which will spawn
50 internal threads, resulting in the operating system seeing 2500 processes competing for the same
50 hardware processors. This "does not end well".

The proper solution would be to get rid of the GIL, not use multi-processing, only use multiple
threads in a pool, and adopt a universal mechanism for running tasks on the threads pool. This will
ensure we have at most 50 threads, used even if parallel tasks spawn nested parallel tasks, and all
memory would be shared so there would be zero-cost in accessing large data structures from different
threads.

You would expect that, two decades after multi-core CPUs became available, this would have been
solved "out of the box" by the current frameworks agreeing to cooperate with each other. However,
somehow this isn't seen as important by the people maintaining these frameworks; in fact, most of
them don't properly handle nested parallelism within their own framework, never mind playing well
with others.

So in practice, while new languages (such as Julia and Rust) provide this out of the box, old
languages (such as Python and C++) are stuck in a swamp. In our case, numpy uses some underlying
parallel threads framework, our own extensions uses OpenMP parallel threads, and we are forced to
use the Python-multi-processing framework itself on top of both, where each of these frameworks is
blind to the rest.

As a crude band-aid, there are ways to force both whatever-numpy-uses and OpenMP to use a specific
number of threads. So, when we use multi-processing, we limit each sub-process to use less internal
threads, such that the total will be at most 50. This is even less optimal, but at least it doesn't
bring the server to its knees trying to deal with a total load of 2500 processes.

A final twist on all this is that hyper-threading is (worse than) useless for heavy compute threads.
We therefore by default only use one thread per physical cores. Python in its infinite wisdom does
not provide a standard way to access the number of physical cores, so we rely on
:py:mod:`metacells.utilities.hardware_info` to guess this amount. This "should" work on Linux,
FreeBSD, Mac OSX and Windows, but YMMV.

.. todo::

    Look into using TBB instead of OpenMP for the inner threads for better efficiency.
'''
import ctypes
import logging
import os
import sys
from multiprocessing import Pool, Value
from threading import current_thread
from typing import Any, Callable, Iterable, Optional, TypeVar

from threadpoolctl import threadpool_limits  # type: ignore

import metacells.extensions as xt  # type: ignore
import metacells.utilities.documentation as utd
import metacells.utilities.hardware_info as uth
import metacells.utilities.timing as utm

__all__ = [
    'set_cpus_count',
    'get_cpus_count',
    'parallel_map',
]


LOG = logging.getLogger(__name__)


CPUS_COUNT = 0

MAIN_PROCESS_PID = os.getpid()
IS_MAIN_PROCESS: Optional[bool] = True

MAP_INDEX = 0
PROCESS_INDEX = 0

PROCESSES_COUNT = 0
NEXT_PROCESS_INDEX = Value(ctypes.c_int32, lock=True)
PARALLEL_FUNCTION: Optional[Callable[[int], Any]] = None


def set_cpus_count(cpus: int) -> None:
    '''
    Set the (maximal) number of CPUs to use in parallel. A value of ``0`` means using all the
    available CPUs. Note that if hyper-threading is enabled, this would be less than the
    number of logical processors in the system. This is intentional as there's no value
    in running multiple heavy computations on hyper-threads of the same CPU.

    The default value is ``0``, that is, using all the CPUs. Override this by setting the
    ``METACELLS_CPUS_COUNT`` environment variable or by invoking this function from the main thread.
    '''
    assert IS_MAIN_PROCESS

    if cpus == 0:
        cpus = uth.hardware_info().get('cpus', os.cpu_count())

    assert cpus > 0

    global CPUS_COUNT
    CPUS_COUNT = cpus

    LOG.debug('CPUS_COUNT: %s', CPUS_COUNT)
    threadpool_limits(limits=CPUS_COUNT)
    xt.set_threads_count(CPUS_COUNT)


if not 'sphinx' in sys.argv[0]:
    set_cpus_count(int(os.environ.get('METACELLS_CPUS_COUNT',
                                      str(os.cpu_count()))))


def get_cpus_count() -> int:
    '''
    Return the number of CPUs we are allowed to use.
    '''
    assert CPUS_COUNT > 0
    return CPUS_COUNT


T = TypeVar('T')


@utd.expand_doc()
def parallel_map(
    function: Callable[[int], T],
    invocations: int,
    *,
    max_cpus: int = 0,
) -> Iterable[T]:
    '''
    Execute ``function``, in parallel, ``invocations`` times. Each invocation is given the
    invocation's index as its single argument.

    For our simple pipelines, only the main process is allowed to execute functions in parallel
    processes, that is, we do not support nested ``parallel_map`` calls.

    This uses :py:func:`get_cpus_count` processes. If ``max_cpus`` (default: {max_cpus}) is
    zero, use all available CPUs. Otherwise, further reduces the number of processes used
    to at most the specified value.

    If this ends up using a single process, runs the function serially. Otherwise, fork new
    processes to execute the function invocations (using ``multiprocessing.Pool.map``).

    The downside is that this is slow. The upside is that each of these processes starts with a
    shared memory copy(-on-write) of the full Python state, that is, all the inputs for the function
    are available "for free".

    .. todo::

        It is currently only possible to invoke :py:func:`parallel_map` from the main application
        thread (that is, it does not nest).
    '''
    assert function.__is_timed__  # type: ignore

    global IS_MAIN_PROCESS
    assert IS_MAIN_PROCESS

    global PROCESSES_COUNT
    PROCESSES_COUNT = min(CPUS_COUNT, invocations)
    if max_cpus != 0:
        assert max_cpus > 0
        PROCESSES_COUNT = min(PROCESSES_COUNT, max_cpus)

    if PROCESSES_COUNT == 1:
        return [function(index) for index in range(invocations)]

    NEXT_PROCESS_INDEX.value = 0

    global PARALLEL_FUNCTION
    assert PARALLEL_FUNCTION is None

    global MAP_INDEX
    MAP_INDEX += 1

    PARALLEL_FUNCTION = function
    IS_MAIN_PROCESS = None
    try:
        utm.flush_timing()
        with utm.timed_step('parallel_map'):
            utm.timed_parameters(index=MAP_INDEX, processes=PROCESSES_COUNT)
            with Pool(PROCESSES_COUNT) as pool:
                return pool.map(_invocation, range(invocations))
    finally:
        IS_MAIN_PROCESS = True
        PARALLEL_FUNCTION = None


def _invocation(index: int) -> Any:
    global IS_MAIN_PROCESS
    if IS_MAIN_PROCESS is None:
        IS_MAIN_PROCESS = os.getpid() == MAIN_PROCESS_PID
        assert not IS_MAIN_PROCESS

        global PROCESS_INDEX
        global NEXT_PROCESS_INDEX
        with NEXT_PROCESS_INDEX:  # type: ignore
            PROCESS_INDEX = NEXT_PROCESS_INDEX.value
            NEXT_PROCESS_INDEX.value += 1

        current_thread().name = '#%s.%s' % (MAP_INDEX, PROCESS_INDEX)

        global CPUS_COUNT
        start_cpu_index = \
            round(CPUS_COUNT * PROCESS_INDEX / PROCESSES_COUNT)
        stop_cpu_index = \
            round(CPUS_COUNT * (PROCESS_INDEX + 1) / PROCESSES_COUNT)
        CPUS_COUNT = stop_cpu_index - start_cpu_index

        assert CPUS_COUNT > 0
        LOG.debug('CPUS_COUNT: %s', CPUS_COUNT)
        threadpool_limits(limits=CPUS_COUNT)
        xt.set_threads_count(CPUS_COUNT)

    assert PARALLEL_FUNCTION is not None
    return PARALLEL_FUNCTION(index)
