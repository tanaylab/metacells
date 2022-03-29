"""
Progress
--------

This used ``tqdm`` to provide a progress bar while computing the metacells.
"""

import logging
from contextlib import contextmanager
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from tqdm import tqdm  # type: ignore

import metacells.utilities.logging as utl

__all__ = [
    "progress_bar",
    "progress_bar_slice",
    "did_progress",
    "has_progress_bar",
    "start_progress_bar",
    "end_progress_bar",
    "start_progress_bar_slice",
    "end_progress_bar_slice",
]


# Parameters for the progress bar.
TQDM_KWARGS: Optional[Dict[str, Any]] = None

# Global progress bar object.
PROGRESS_BAR: Optional[tqdm] = None

# The log level when the progress bar was activated.
PROGRESS_LOG_LEVEL = logging.INFO

# The current progress position.
PROGRESS_POSITION = 0

# The current progress bar (slice) final position.
PROGRESS_END = 0

#: The size of the current progress bar (slice)
PROGRESS_SIZE = 0


def has_progress_bar() -> bool:
    """
    Return whether there is an active progress bar.
    """
    return PROGRESS_BAR is not None or TQDM_KWARGS is not None


@contextmanager
def progress_bar(**tqdm_kwargs: Any) -> Any:
    """
    Run some code with a ``tqdm`` progress bar.

    ..note::

        When a progress bar is active, logging is restricted to warnings and errors.
    """
    start_progress_bar(**tqdm_kwargs)
    try:
        result = yield None
        return result
    finally:
        end_progress_bar()


def start_progress_bar(**tqdm_kwargs: Any) -> Any:
    """
    Create a progress bar (but do not show it yet).
    """
    global TQDM_KWARGS
    global PROGRESS_POSITION
    global PROGRESS_END
    global PROGRESS_SIZE
    global PROGRESS_LOG_LEVEL

    assert PROGRESS_BAR is None
    assert TQDM_KWARGS is None

    TQDM_KWARGS = tqdm_kwargs.copy()
    if "bar_format" not in TQDM_KWARGS:
        TQDM_KWARGS["bar_format"] = "{l_bar}{bar}[{elapsed}]"  # NOT F-STRING

    PROGRESS_POSITION = 0
    PROGRESS_END = int(1e15)
    PROGRESS_SIZE = PROGRESS_END
    PROGRESS_LOG_LEVEL = utl.logger().level
    utl.logger().setLevel(logging.WARNING)


def end_progress_bar() -> None:
    """
    End an active progress bar.
    """
    global PROGRESS_BAR
    if PROGRESS_BAR is not None:
        PROGRESS_BAR.close()
        utl.logger().setLevel(PROGRESS_LOG_LEVEL)
        PROGRESS_BAR = None


@contextmanager
def progress_bar_slice(fraction: Optional[float]) -> Any:
    """
    Run some code which will use a slice of the current progress bar.

    This can be nested to split the overall progress bar into smaller and smaller parts to represent a tree of
    computations.

    If ``fraction`` is None and there is no active progress bar, simply runs the code.
    """
    _show_progress_bar()
    assert (fraction is not None) == has_progress_bar()
    if fraction is None:
        result = yield None
        return result

    old_state = start_progress_bar_slice(fraction)
    try:
        result = yield None
        return result
    finally:
        end_progress_bar_slice(old_state)


def start_progress_bar_slice(fraction: float) -> Tuple[int, int]:
    """
    Start a nested slice of the overall progress bar.

    Returned the captured state that needs to be passed to ``end_progress_bar_slice``.
    """
    assert PROGRESS_BAR is not None
    assert TQDM_KWARGS is None
    assert 0.0 < fraction < 1.0

    global PROGRESS_END
    global PROGRESS_SIZE

    old_progress_size = PROGRESS_SIZE
    PROGRESS_SIZE = int(round(PROGRESS_SIZE * fraction))
    assert 0 < PROGRESS_SIZE < old_progress_size

    old_progress_end = PROGRESS_END
    PROGRESS_END = PROGRESS_POSITION + PROGRESS_SIZE
    assert PROGRESS_END <= old_progress_end

    return old_progress_size, old_progress_end


def end_progress_bar_slice(old_state: Tuple[int, int]) -> None:
    """
    End a nested slice of the overall progress bar.

    This moves the progress bar position to the end of the slice regardless of reported progress within it.
    """
    old_progress_size, old_progress_end = old_state

    global PROGRESS_POSITION
    global PROGRESS_END
    global PROGRESS_SIZE

    assert PROGRESS_BAR is not None
    assert TQDM_KWARGS is None
    assert PROGRESS_POSITION <= PROGRESS_END

    remaining = PROGRESS_END - PROGRESS_POSITION
    if remaining > 0:
        PROGRESS_BAR.update(remaining)
    PROGRESS_POSITION = PROGRESS_END
    PROGRESS_SIZE = old_progress_size
    PROGRESS_END = old_progress_end


def did_progress(fraction: float) -> Any:
    """
    Report progress of some fraction of the current (slice of) progress bar.
    """
    _show_progress_bar()
    assert 0 < fraction <= 1.0
    assert PROGRESS_BAR is not None
    assert TQDM_KWARGS is None

    global PROGRESS_POSITION

    step = int(round(PROGRESS_SIZE * fraction))
    step = min(step, PROGRESS_END - PROGRESS_POSITION)

    if step > 0:
        PROGRESS_BAR.update(step)
        PROGRESS_POSITION += step


def _show_progress_bar() -> None:
    global PROGRESS_BAR
    global TQDM_KWARGS
    if PROGRESS_BAR is None and TQDM_KWARGS is not None:
        PROGRESS_BAR = tqdm(total=PROGRESS_SIZE, **TQDM_KWARGS)  # pylint: disable=not-a-mapping
        TQDM_KWARGS = None
