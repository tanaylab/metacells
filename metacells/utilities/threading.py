'''
Utilities for multi-threaded code.

.. todo::

    While multi-threading does seem to work reasonably well (given the limitations of the GIL), it
    seems that direct C++ re-implementations of common builtin operations (such as ``bincount``) are
    not as efficient as the builtin implementations, so the gains from a parallel version are
    negated. This requires further investigation.
'''
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import current_thread
from typing import (Any, Callable, Dict, Iterable, Iterator, KeysView, List,
                    Optional, Tuple, Union)

import numpy as np  # type: ignore
from readerwriterlock import rwlock

import metacells.utilities.timing as timed
from metacells.utilities.documentation import expand_doc

__all__ = [
    'THREADS_COUNT',
    'EXECUTOR',
    'parallel_map',
    'parallel_for',
    'parallel_collect',
    'SharedStorage',
]

#: The number of threads to use. Override this by setting the ``METACELLS_THREADS_COUNT``
#: environment variable: -1 uses half of the available processors to combat hyper-threading, 0 uses
#: all the available processors, otherwise the value is the number of threads to use.
THREADS_COUNT = 0

#: The executor to use for parallel tasks (only set if using multiple threads).
EXECUTOR: Optional[ThreadPoolExecutor] = None

if not 'sphinx' in sys.argv[0]:
    THREADS_COUNT = \
        int(os.environ.get('METACELLS_THREADS_COUNT', str(os.cpu_count())))
    if THREADS_COUNT == -1:
        THREADS_COUNT = (os.cpu_count() or 1) // 2
    elif THREADS_COUNT == 0:
        THREADS_COUNT = os.cpu_count() or 1

    assert THREADS_COUNT > 0

    if THREADS_COUNT > 1:
        EXECUTOR = ThreadPoolExecutor(THREADS_COUNT)


@expand_doc()
def parallel_map(
    function: Callable[[Union[int, range]], Any],
    invocations_count: int,
    *,
    batches_per_thread: Optional[int] = 3,
) -> Iterable[Any]:
    '''
    Execute ``function``, in parallel, ``invocations_count`` times. Each invocation is given the
    invocation's index as its single argument.

    For efficiently processing a large number of invocations, the invocations are grouped to batches
    such that on average each thread will execute no more than ``batches_per_thread``. In general,
    there is an inherent trade-off where a higher number of batches per thread increases scheduling
    overhead but reduces sensitivity for variance between execution time of function invocations.
    The default value of {batches_per_thread} is meant to provide a reasonable compromise.

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
        return function(range(invocations_count))

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
        _analyze_loop(invocations_count, batches_per_thread)

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


def parallel_for(
    function: Callable,
    invocations_count: int,
    *,
    batches_per_thread: Optional[int] = 4,
) -> None:
    '''
    Similar to ``parallel_map`` except that the return value of ``function`` is ignored. This avoid
    the inefficiencies of collecting the results (especially when batching is used). It is often
    used when the ``function`` invocations write some result(s) into some shared-memory array(s).
    '''
    if EXECUTOR is None:
        if batches_per_thread is None:
            for _ in map(function, range(invocations_count)):
                pass
        else:
            function(range(invocations_count))
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
        _analyze_loop(invocations_count, batches_per_thread)

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


def parallel_collect(
    compute: Callable,
    merge: Callable,
    invocations_count: int,
    *,
    batches_per_thread: Optional[int] = 4,
) -> str:
    '''
    Similar to ``parallel_for``, except it is assumed that each invocation of ``compute`` updates
    some private per-thread variable(s) in some storage, and ``merge(from_thread,
    into_thread)`` will merge the data from the private storage variable(s) of one thread into
    the private storage of another.

    Returns the name of the thread which executed the final ``merge`` operations so the caller can
    access the final collected data.

    .. note::
        It is safe for ``merge`` to use
        :py:meth:`metacells.utilities.threading.SharedStorage.get_private`
        :py:meth:`metacells.utilities.threading.SharedStorage.set_private` to access the data
        collected in the ``from_thread`` and ``into_thread``.

    .. todo::

        The :py:func:`parallel_collect` implementation first performs all the ``compute``
        invocations (in parallel) and then all the ``merge`` invocations (also in parallel). A more
        efficient implementation would invoke ``merge`` earlier in parallel to some ``compute``
        calls.
    '''
    if EXECUTOR is None:
        parallel_for(compute, invocations_count,
                     batches_per_thread=batches_per_thread)
        return current_thread().name

    shared_storage = SharedStorage()

    shared_storage.set_shared('pending', set())

    def compute_function(*args: Any) -> None:
        compute(*args)
        thread = current_thread().name
        with shared_storage.read_shared('pending') as pending:
            if thread in pending:
                return
        with shared_storage.write_shared('pending') as pending:
            pending.add(thread)

    parallel_for(compute_function, invocations_count,
                 batches_per_thread=batches_per_thread)

    pending = shared_storage.get_shared('pending')

    if len(pending) > 1:
        def merge_function(_index: int) -> None:
            into_thread: Optional[str] = None
            while True:
                with shared_storage.write_shared('pending') as pending:
                    if into_thread is None:
                        if len(pending) == 0:
                            return
                        into_thread = pending.pop()

                    if len(pending) == 0:
                        pending.add(into_thread)
                        return

                    from_thread = pending.pop()

                merge(from_thread, into_thread)

        parallel_for(merge_function, len(pending) - 1, batches_per_thread=None)

    assert len(pending) == 1
    return pending.pop()


def _analyze_loop(
    invocations_count: int,
    batches_per_thread: int,
) -> Tuple[int, float]:
    batches_count = THREADS_COUNT * batches_per_thread
    batch_size = invocations_count / batches_count

    if batch_size <= 1.0:
        return invocations_count, 1.0

    assert np.isclose(batches_count * batch_size, invocations_count)
    return batches_count, batch_size


class SharedStorage:
    '''
    Shared storage for parallel tasks.

    .. note::

        The implementation assumes that Python dictionary operations are atomic, in the sense that
        if different threads read/write different keys in the same shared dictionary, then
        everything will work as expected. This seems to be the case (e.g., see
        stackoverflow question 1312331_.)

        .. _1312331: https://stackoverflow.com/questions/1312331/using-a-global-dictionary-with-threads-in-python
    '''

    def __init__(self) -> None:
        '''
        Create an empty storage for parallel tasks.
        '''
        self._shared: Dict[str, Tuple[rwlock.RWLockRead, Any]] = {}
        self._private: Dict[str, Dict[str, Any]] = {}

    def set_shared(self, name: str, value: Any) -> None:
        '''
        Set the ``value`` of a shared ``named`` data.

        Shared data is accessible to all threads.

        .. note::

            This is meant to be invoked from the main thread before sending the storage to be used
            by multiple threads. As such, it assumes no other thread is accessing the storage at the
            same time.
        '''
        self._shared[name] = (rwlock.RWLockRead(), value)

    @contextmanager
    def read_shared(self, name: str) -> Iterator[Any]:
        '''
        Safe(-ish) read-only access to shared data by its ``name``.

        .. note::

            This is meant to be used by threads who want to ensure that no other thread is modifying
            the data. It will be safe against modification by ``with write_shared``, but not against
            access using ``get_shared``.
        '''
        lock, value = self._shared[name]
        with lock.gen_rlock():
            yield value

    @contextmanager
    def write_shared(self, name: str) -> Iterator[Any]:
        '''
        Safe(-ish) exclusive read-write access to shared data by its ``name``.

        .. note::

            This is meant to be used by threads who want to modify the data even if other threads
            might be reading it. It will be safe against read-only-access by ``with read_shared``,
            but not against access using ``get_shared``.
        '''
        lock, value = self._shared[name]
        with lock.gen_wlock():
            yield value

    def get_shared(self, name: str) -> Any:
        '''
        Unsafe access to shared data (which must exist) by its ``name``.

        .. note::

            This is meant to be used by the threads if all only read the data, or if each thread
            safely modified different parts of the data (e.g. writing to different array entries),
            or the threads otherwise somehow coordinate safe shared access to the data.
        '''
        _lock, value = self._shared[name]
        return value

    def set_private(self, name: str, value: Any, *, thread: Optional[str] = None) -> None:
        '''
        Safe(-ish) set the ``value`` (which must not be ``None``) of private data for the a thread
        by its ``name``.

        Private data will have a copy per thread. If the ``thread`` is not specified, sets the value
        for the current thread.

        .. note::

            This is intended to be used from within each thread to set its own data. As such, it
            assumes that no other thread is accessing this thread's private data at the same time.

            Likewise, if ``thread`` is specified, then the caller is responsible for ensuring that
            multiple threads are not accessing the same data at the same time.
        '''
        assert value is not None
        if thread is None:
            thread = current_thread().name

        values = self._private.get(thread)
        if values is None:
            values = self._private[thread] = {}

        values[name] = value

    def get_private(  #
        self,
        name: str,
        *,
        thread: Optional[str] = None,
        make: Optional[Callable[[], Any]] = None,
        update: Optional[Callable[[Any], Any]] = None,
    ) -> Any:
        '''
        Safe(-ish) access to the private data of a ``thread`` by its ``name``.

        Private data will have a copy per thread. If the ``thread`` is not specified, sets the value
        for the current thread.

        If the value was not set yet, this is an error, unless ``make`` is specified which will be
        invoked to return the initial private value (which must not be ``None``).

        If the value has been previously set, and ``update`` is specified, then ``update(value)`` is
        invoked and set as the new ``value``, and returned.

        .. note::

            This is intended to be used from within each thread to get its own data. As such, it
            assumes that no other thread is accessing this thread's private data at the same time.

            Likewise, if ``thread`` is specified, then the caller is responsible for ensuring that
            multiple threads are not accessing the same data at the same time.
        '''
        if thread is None:
            thread = current_thread().name

        values = self._private.get(thread)
        if values is None:
            values = self._private[thread] = {}

        if make is None:
            value = values[name]
        else:
            value = values.get(name)
            if value is None:
                value = values[name] = make()
                assert value is not None
                return value

        if update is not None:
            value = values[name] = update(value)
            assert value is not None

        return value

    def get_threads(self) -> KeysView[str]:
        '''
        Return the names of the threads who created private data.

        .. note::

            This is intended to be used from the main thread after all the other threads are done
            executing, in order to collect the data each collected. As such, it assumes that no
            other thread is accessing the storage at the same time.
        '''
        return self._private.keys()
