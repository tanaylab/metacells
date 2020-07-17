'''
Utilities for multi-threaded code.
'''
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np  # type: ignore

import metacells.utilities.timing as timed
from metacells.utilities.documentation import expand_doc

__all__ = [
    'parallel_map',
    'parallel_for',
]


THREADS_COUNT = int(os.environ.get(
    'METACELLS_THREADS_COUNT', str(os.cpu_count())))
if THREADS_COUNT == 0:
    THREADS_COUNT = os.cpu_count() or 1

EXECUTOR = ThreadPoolExecutor(THREADS_COUNT) if THREADS_COUNT > 1 else None


@expand_doc()
def parallel_map(
    function: Callable[[Union[int, range]], Any],
    invocations_count: int,
    *,
    batches_per_thread: Optional[int] = 4,
    minimal_invocations_per_batch: int = 1,
) -> Iterable[Any]:
    '''
    Execute ``function``, in parallel, ``invocations_count`` times. Each invocation is given the
    invocation's index as its single argument.

    For efficiently processing a large number of invocations, the invocations are grouped to batches
    such that on average each thread will execute no more than ``batches_per_thread``. In general,
    there is an inherent trade-off where a higher number of batches per thread increases scheduling
    overhead but reduces sensitivity for variance between execution time of function invocations.
    The default value of {batches_per_thread} is meant to provide a reasonable compromise.

    If ``minimal_invocations_per_batch`` is specified, it may reduce the number of batches,
    ensuring that a lower invocation count does not incur excessive scheduling overhead. Picking a
    good value here requires profiling, and taking into account the amount of work in each
    invocation. A good way to collect the needed information is to use ``timing.parameters``.

    When batching is used, the ``function`` is invoked with a range of invocation indices, and
    should return a list of results, one per invocation index. This allows the function to perform
    more efficient inner loop on the batch of invocations.

    If ``batches_per_thread`` is ``None``, then the function is given just a single invocation index
    and is expected to return just a single result. This allows for simpler invocation of
    heavy-weight functions (that possible internally use parallel operations).

    .. note::

        Since python uses the notorious GIL, it does not benefit as much from multiple threads as
        could be hoped. Luckily, computational libraries such as ``numpy`` perform many actions
        without requiring the GIL, so parallel execution does provide benefits for code which mostly
        invokes such computational library functions.
    '''
    if EXECUTOR is None:
        if batches_per_thread is None:
            return map(function, range(invocations_count))
        return _flatten(map(function, [range(invocations_count)]))

    step_timing = timed.current_step()

    if batches_per_thread is None:
        if step_timing is None:
            timed_function = function
        else:
            def timed_function(index: int) -> None:  # type: ignore
                with timed.step(step_timing):  # type: ignore
                    function(index)

        return list(EXECUTOR.map(timed_function, range(invocations_count)))

    batches_count, batch_size = \
        _analyze_loop(invocations_count, batches_per_thread,
                      minimal_invocations_per_batch)

    timed.parameters('invocations_count', invocations_count,
                     'batches_count', batches_count,
                     'batch_size', batch_size)

    if batches_count <= 1:
        return function(range(invocations_count))

    if step_timing is None:
        def batch_function(batch_index: int) -> List[Any]:
            start = round(batch_index * batch_size)
            stop = round((batch_index + 1) * batch_size)
            indices = range(start, stop)
            results_of_chunk = function(indices)
            assert len(results_of_chunk) == len(indices)
            return results_of_chunk

    else:
        def batch_function(batch_index: int) -> List[Any]:
            with timed.step(step_timing):  # type: ignore
                start = round(batch_index * batch_size)
                stop = round((batch_index + 1) * batch_size)
                indices = range(start, stop)
                results_of_chunk = function(indices)
                assert len(results_of_chunk) == len(indices)
                return results_of_chunk

    return _flatten(EXECUTOR.map(batch_function, range(batches_count)))


def _flatten(list_of_lists: Iterable[Iterable[Any]]) -> List[Any]:
    return [item for list_of_items in list_of_lists for item in list_of_items]


@expand_doc()
def parallel_for(
    function: Callable,
    invocations_count: int,
    *,
    batches_per_thread: Optional[int] = 4,
    minimal_invocations_per_batch: int = 1,
) -> None:
    '''
    Similar to ``parallel_map`` except that the return value of ``function`` is ignored. This avoid
    the inefficiencies of collecting the results (especially when batching is used). It is often
    used when the ``function`` invocations write some result(s) into some shared-memory array(s).
    '''
    if EXECUTOR is None:
        if batches_per_thread is None:
            map(function, range(invocations_count))
        else:
            map(function, [range(invocations_count)])
        return

    step_timing = timed.current_step()

    if batches_per_thread is None:
        if step_timing is None:
            timed_function = function
        else:
            def timed_function(index: int) -> None:  # type: ignore
                with timed.step(step_timing):  # type: ignore
                    function(index)

        for _ in EXECUTOR.map(timed_function, range(invocations_count)):
            pass
        return

    batches_count, batch_size = \
        _analyze_loop(invocations_count, batches_per_thread,
                      minimal_invocations_per_batch)

    timed.parameters('invocations_count', invocations_count,
                     'batches_count', batches_count,
                     'batch_size', batch_size)

    if batches_count <= 1:
        function(range(invocations_count))
        return

    if step_timing is None:
        def batch_function(batch_index: int) -> None:
            start = round(batch_index * batch_size)
            stop = round((batch_index + 1) * batch_size)
            indices = range(start, stop)
            function(indices)

    else:
        def batch_function(batch_index: int) -> None:
            with timed.step(step_timing):  # type: ignore
                start = round(batch_index * batch_size)
                stop = round((batch_index + 1) * batch_size)
                indices = range(start, stop)
                function(indices)

    for _ in EXECUTOR.map(batch_function, range(batches_count)):
        pass


def _analyze_loop(
    invocations_count: int,
    batches_per_thread: int,
    minimal_invocations_per_batch: int,
) -> Tuple[int, float]:
    batches_count = THREADS_COUNT * batches_per_thread
    batch_size = invocations_count / batches_count

    if batch_size < minimal_invocations_per_batch:
        batches_count = \
            max(int(invocations_count / minimal_invocations_per_batch), 1)
        batch_size = invocations_count / batches_count

    if batch_size <= 1.0:
        return invocations_count, 1.0

    assert np.isclose(batches_count * batch_size, invocations_count)
    return batches_count, batch_size
