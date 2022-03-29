"""
Parallel
--------

Due to the notorious GIL, using multiple Python threads is essentially useless. This leaves us with
two options for using multiple processors, which is mandatory for reasonable performance on the
large data sets we work on:

* Use multiple threads in the internal C++ implementation of some Python functions; this is done by both numpy and the
  C++ extension functions provided by this package, and works even for reasonably small sized work, such as sorting each
  of the rows of a large matrix.

* Use Python multi-processing. This is costly and works only for large sized work, such as computing metacells for
  different piles.

Each of these two approaches works tolerably well on its own, even though both are sub-optimal. The
problem starts when we want to combine them. Consider a server with 50 processors. Invoking
``corrcoef`` on a large matrix will use them all. This is great if one computes metacells for a
single pile. Suppose, however, you want to compute metacells for 50 piles, and do so using
multi-processing. Each and every of the 50 sub-processes will invoke ``corcoeff`` which will spawn
50 internal threads, resulting in the operating system seeing 2500 processes competing for the same
50 hardware processors. "This does not end well."

You would expect that, two decades after multi-core systems became available, this would have been
solved "out of the box" by the parallel frameworks (Python, OpenMP, TBB, etc.) all agreeing to
cooperate with each other. However, somehow this isn't seen as important by the people maintaining
these frameworks; in fact, most of them don't properly handle nested parallelism within their own
framework, never mind playing well with others.

So in practice, while languages built for parallelism (such as Julia and Rust) deal well with nested
parallel construct, using a mixture of older serial languages (such as Python and C++) puts us in a
swamp, and "you can't build a castel in a swamp". In our case, numpy uses some underlying parallel
threads framework, our own extensions uses OpenMP parallel threads, and we are forced to use the
Python-multi-processing framework itself on top of both, and each of these frameworks is blind to
the others.

As a crude band-aid, we force both whatever-numpy-uses and OpenMP to use a specific number of
threads. So, when we use multi-processing, we limit each sub-process to use less internal threads,
such that the total will be at most 50. This very sub-optimal, but at least it doesn't bring the
server to its knees trying to deal with a total load of 2500 processes.

A final twist on all this is that hyper-threading is (worse than) useless for heavy compute threads.
We therefore by default only use one thread per physical cores. We get the number pf physical cores
using the ``psutil`` package.

.. todo::

    Re-implement all the package in a single language more suitable for scientific computing. Julia
    is looking like a good combination of convenience and performance...
"""
import ctypes
import os
import sys
from multiprocessing import Value
from multiprocessing import get_context
from threading import current_thread
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar

import psutil  # type: ignore
from threadpoolctl import threadpool_limits  # type: ignore

import metacells.utilities.documentation as utd
import metacells.utilities.logging as utl
import metacells.utilities.progress as utp
import metacells.utilities.timing as utm

if "sphinx" not in sys.argv[0]:
    import metacells.extensions as xt  # type: ignore

__all__ = [
    "is_main_process",
    "set_processors_count",
    "get_processors_count",
    "parallel_map",
]


PROCESSORS_COUNT = 0

MAIN_PROCESS_PID = os.getpid()

IS_MAIN_PROCESS: Optional[bool] = True

MAP_INDEX = 0
PROCESS_INDEX = 0

PROCESSES_COUNT = 0
NEXT_PROCESS_INDEX = Value(ctypes.c_int32, lock=True)
PARALLEL_FUNCTION: Optional[Callable[[int], Any]] = None


def is_main_process() -> bool:
    """
    Return whether this is the main process, as opposed to a sub-process spawned by
    :py:func:`parallel_map`.
    """
    return bool(IS_MAIN_PROCESS)


def set_processors_count(processors: int) -> None:
    """
    Set the (maximal) number of processors to use in parallel.

    The default value of ``0`` means using all the available physical processors. Note that if
    hyper-threading is enabled, this would be less than (typically half of) the number of logical
    processors in the system. This is intentional, as there's no value - actually, negative
    value - in running multiple heavy computations on hyper-threads of the same physical processor.

    Otherwise, the value is the actual (positive) number of processors to use. Override this by
    setting the ``METACELLS_PROCESSORS_COUNT`` environment variable or by invoking this function
    from the main thread.
    """
    assert IS_MAIN_PROCESS

    if processors == 0:
        processors = psutil.cpu_count(logical=False)

    assert processors > 0

    global PROCESSORS_COUNT
    PROCESSORS_COUNT = processors

    threadpool_limits(limits=PROCESSORS_COUNT)
    xt.set_threads_count(PROCESSORS_COUNT)


if "sphinx" not in sys.argv[0]:
    set_processors_count(int(os.environ.get("METACELLS_PROCESSORS_COUNT", "0")))


def get_processors_count() -> int:
    """
    Return the number of PROCESSORs we are allowed to use.
    """
    assert PROCESSORS_COUNT > 0
    return PROCESSORS_COUNT


T = TypeVar("T")


@utd.expand_doc()
def parallel_map(
    function: Callable[[int], T],
    invocations: int,
    *,
    max_processors: int = 0,
    hide_from_progress_bar: bool = False,
) -> List[T]:
    """
    Execute ``function``, in parallel, ``invocations`` times. Each invocation is given the invocation's index as its
    single argument.

    For our simple pipelines, only the main process is allowed to execute functions in parallel processes, that is, we
    do not support nested ``parallel_map`` calls.

    This uses :py:func:`get_processors_count` processes. If ``max_processors`` (default: {max_processors}) is zero, use
    all available processors. Otherwise, further reduces the number of processes used to at most the specified value.

    If this ends up using a single process, runs the function serially. Otherwise, fork new processes to execute the
    function invocations (using ``multiprocessing.get_context('fork').Pool.map``).

    The downside is that this is slow, and you need to set up **mutable** shared memory (e.g. for large results) in
    advance. The upside is that each of these processes starts with a shared memory copy(-on-write) of the full Python
    state, that is, all the inputs for the function are available "for free".

    If a progress bar is active at the time of invoking ``parallel_map``, and ``hide_from_progress_bar`` is not set,
    then it is assumed the parallel map will cover all the current (slice of) the progress bar, and it is reported into
    it in increments of ``1/invocations``.

    .. todo::

        It is currently only possible to invoke :py:func:`parallel_map` from the main application thread (that is, it
        does not nest).
    """
    assert function.__is_timed__  # type: ignore

    global IS_MAIN_PROCESS
    assert IS_MAIN_PROCESS

    global PROCESSES_COUNT
    PROCESSES_COUNT = min(PROCESSORS_COUNT, invocations)
    if max_processors != 0:
        assert max_processors > 0
        PROCESSES_COUNT = min(PROCESSES_COUNT, max_processors)

    if PROCESSES_COUNT == 1:
        return [function(index) for index in range(invocations)]

    NEXT_PROCESS_INDEX.value = 0  # type: ignore

    global PARALLEL_FUNCTION
    assert PARALLEL_FUNCTION is None

    global MAP_INDEX
    MAP_INDEX += 1

    PARALLEL_FUNCTION = function
    IS_MAIN_PROCESS = None
    try:
        results: List[Optional[T]] = [None] * invocations
        utm.flush_timing()
        with utm.timed_step("parallel_map"):
            utm.timed_parameters(index=MAP_INDEX, processes=PROCESSES_COUNT)
            with get_context("fork").Pool(PROCESSES_COUNT) as pool:
                for index, result in pool.imap_unordered(_invocation, range(invocations)):
                    if utp.has_progress_bar() and not hide_from_progress_bar:
                        utp.did_progress(1 / invocations)
                    results[index] = result
        return results  # type: ignore
    finally:
        IS_MAIN_PROCESS = True
        PARALLEL_FUNCTION = None


def _invocation(index: int) -> Tuple[int, Any]:
    global IS_MAIN_PROCESS
    if IS_MAIN_PROCESS is None:
        IS_MAIN_PROCESS = os.getpid() == MAIN_PROCESS_PID
        assert not IS_MAIN_PROCESS

        global PROCESS_INDEX
        with NEXT_PROCESS_INDEX:
            PROCESS_INDEX = NEXT_PROCESS_INDEX.value  # type: ignore
            NEXT_PROCESS_INDEX.value += 1  # type: ignore

        current_thread().name = f"#{MAP_INDEX}.{PROCESS_INDEX}"
        utm.in_parallel_map(MAP_INDEX, PROCESS_INDEX)

        global PROCESSORS_COUNT
        start_processor_index = int(round(PROCESSORS_COUNT * PROCESS_INDEX / PROCESSES_COUNT))
        stop_processor_index = int(round(PROCESSORS_COUNT * (PROCESS_INDEX + 1) / PROCESSES_COUNT))
        PROCESSORS_COUNT = stop_processor_index - start_processor_index

        assert PROCESSORS_COUNT > 0
        utl.logger().debug("PROCESSORS: %s", PROCESSORS_COUNT)
        threadpool_limits(limits=PROCESSORS_COUNT)
        xt.set_threads_count(PROCESSORS_COUNT)

    assert PARALLEL_FUNCTION is not None
    return index, PARALLEL_FUNCTION(index)
