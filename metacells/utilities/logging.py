'''
Logging
-------

This provides a useful formatter which includes high-resolution time and thread names, and a set of
utility functions for effective logging of operations on annotation data.

Collection of log messages is mostly automated by wrapping relevant function calls and tracing the
setting and getting of data via the :py:mod:`metacells.utilities.annotation` accessors, with the
occasional explicit logging of a notable intermediate calculated value via :py:func:`log_calc`.

The ging is picking the correct level for each log message. This module provides the following log
levels which hopefully provide the end user with a reasonable amount of control:

* ``INFO`` will log only setting of the final results as annotations within the top-level
  ``AnnData`` object(s).

* ``STEP`` will also log the top-level algorithm step(s), which give a very basic insight into
  what was executed.

* ``PARAM`` will also log the parameters of these steps, which may be important when tuning the
  behavior of the system for different data sets.

* ``CALC`` will also log notable intermediate calculated results, which again may be important
  when tuning the behavior of the system for different data sets.

* ``DEBUG`` pulls all the stops and logs all the above, not only for the top-level steps, but
  also for the nested processing steps. This results in a rather large log file (especially
  for the recursive divide-and-conquer algorithm). You don't need this except for when
  you really need this.

To achieve this, we track for each ``AnnData`` whether it is a top-level (user visible) or a
temporary data object, and whether we are inside a top-level (user invoked) or a nested operation.
Accessing top-level data and invoking top-level operations is logged at the coarse logging levels,
anything else is logged at the ``DEBUG`` level.

To improve the log messages, we allow each ``AnnData`` object to have an optional name for logging
(see :py:func:`metacells.utilities.annotation.set_name` and
:py:func:`metacells.utilities.annotation.get_name`). Whenever a temporary ``AnnData`` data is
created, its name is extended by some descriptive suffix, so we get names like
``full.clean.feature`` to describe the feature data extracted out of the clean data extracted out of
the full data.
'''

import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from inspect import Parameter, signature
from logging import (DEBUG, INFO, Formatter, Logger, LogRecord, StreamHandler,
                     getLogger, setLoggerClass)
from multiprocessing import Lock
from threading import current_thread
from typing import (IO, Any, Callable, Dict, Iterator, List, Optional, Tuple,
                    TypeVar, Union)

import numpy as np
import pandas as pd  # type: ignore
from anndata import AnnData

import metacells.utilities.documentation as utd
import metacells.utilities.parallel as utp
import metacells.utilities.typing as utt

__all__ = [
    'setup_logger',
    'logger',
    'CALC',
    'STEP',
    'PARAM',
    'logged',
    'top_level',
    'log_return',
    'logging_calc',
    'log_calc',
    'log_step',
    'incremental',
    'done_incrementals',
    'cancel_incrementals',
    'log_set',
    'log_get',
    'sizes_description',
    'fractions_description',
    'groups_description',
    'mask_description',
    'ratio_description',
    'progress_description',
    'fraction_description',
    'fold_description',
]


class SynchronizedLogger(Logger):
    '''
    A logger that synchronizes between the sub-processes.

    This ensures logging does not get garbled when when using multi-processing
    (e.g., using :py:func:`metacells.utilities.parallel.parallel_map`).
    '''

    LOCK = Lock()

    def _log(self, *args: Any, **kwargs: Any) -> Any:  # pylint: disable=signature-differs
        with SynchronizedLogger.LOCK:
            super()._log(*args, **kwargs)


class LoggingFormatter(Formatter):
    '''
    A formatter that uses a decimal point for milliseconds.
    '''

    def formatTime(self, record: Any, datefmt: Optional[str] = None) -> str:
        '''
        Format the time.
        '''
        record_datetime = datetime.fromtimestamp(record.created)
        if datefmt is not None:
            assert False
            return record_datetime.strftime(datefmt)

        seconds = record_datetime.strftime('%Y-%m-%d %H:%M:%S')
        msecs = round(record.msecs)
        return f'{seconds}.{msecs:03d}'


#: The log level for tracing processing steps.
STEP = (1 * DEBUG + 3 * INFO) // 4

#: The log level for tracing parameters.
PARAM = (2 * DEBUG + 2 * INFO) // 4

#: The log level for tracing intermediate calculations.
CALC = (3 * DEBUG + 1 * INFO) // 4

logging.addLevelName(STEP, 'STEP')
logging.addLevelName(PARAM, 'PARAM')
logging.addLevelName(CALC, 'CALC')


class ShortLoggingFormatter(LoggingFormatter):
    '''
    Provide short level names.
    '''

    #: Map the long level names to the fixed-width short level names.
    SHORT_LEVEL_NAMES = dict(CRITICAL='CRT', CRT='CRT',
                             ERROR='ERR', ERR='ERR',
                             WARNING='WRN', WRN='WRN',
                             INFO='INF', INF='INF',
                             STEP='STP', STP='STP',
                             PARAM='PRM', PRM='PRM',
                             HINT='PRM', HNT='PRM',
                             CALC='CLC', CLC='CLC',
                             DEBUG='DBG', DBG='DBG',
                             NOTSET='NOT', NOT='NOT')

    def format(self, record: LogRecord) -> Any:
        record.levelname = self.SHORT_LEVEL_NAMES[record.levelname]
        return LoggingFormatter.format(self, record)


# ' Global logger object.
LOG: Optional[Logger] = None


@utd.expand_doc()
def setup_logger(
    *,
    level: int = logging.INFO,
    to: IO = sys.stderr,
    time: bool = False,
    process: Optional[bool] = None,
    name: Optional[str] = None,
    long_level_names: Optional[bool] = None,
) -> Logger:
    '''
    Setup the global :py:func:`logger`.

    .. note::

        A second call will fail as the logger will already be set up.

    If ``level`` is not specified, only ``INFO`` messages (setting values in the annotated data)
    will be logged.

    If ``to`` is not specified, the output is sent to ``sys.stderr``.

    If ``time`` (default: {time}), include a millisecond-resolution timestamp in each message.

    If ``name`` (default: {name}) is specified, it is added to each message.

    If ``process`` (default: {process}), include the (sub-)process index in each message. The name
    of the main process (thread) is replaced to ``#0`` to make it more compatible with the
    sub-process names (``#<map-index>.<sub-process-index>``).

    If ``process`` is ``None``, and if the logging level is higher than ``INFO``, and
    :py:func:`metacells.utilities.parallel.get_processors_count` is greater than one, then
    ``process`` is set - that is, it will be set if we expect to see log messages from multiple
    sub-processes.

    Logging from multiple sub-processes (e.g., using (e.g., using
    :py:func:`metacells.utilities.parallel.parallel_map`) will synchronize using a global lock so
    messages will not get garbled.

    If ``long_level_names`` (default: {long_level_names}), includes the log level in each message.
    If is ``False``, the log level names are shortened to three characters, for consistent
    formatting of indented (nested) log messages. If it is ``None``, no level names are logged at
    all.
    '''
    assert utp.is_main_process()

    global LOG
    assert LOG is None

    if long_level_names is not None:
        log_format = '%(levelname)s - %(message)s'
    else:
        log_format = '%(message)s'

    if process is None:
        process = level < logging.INFO and utp.get_processors_count() > 1

    if process:
        log_format = '%(threadName)s - ' + log_format
        current_thread().name = '#0'

    if name is not None:
        log_format = name + ' - ' + log_format
    if time:
        log_format = '%(asctime)s - ' + log_format

    handler = StreamHandler(to)
    if long_level_names == False:  # pylint: disable=singleton-comparison
        handler.setFormatter(ShortLoggingFormatter(log_format))
    else:
        handler.setFormatter(LoggingFormatter(log_format))
    setLoggerClass(SynchronizedLogger)
    LOG = getLogger('metacells')
    LOG.addHandler(handler)
    LOG.setLevel(level)

    LOG.debug('PROCESSORS: %s', utp.get_processors_count())
    return LOG


def logger() -> Logger:
    '''
    Access the global logger.

    If :py:func:`setup_logger` has not been called yet, this will call it using the default flags.
    You should therefore call :py:func:`setup_logger` as early as possible to ensure you don't end
    up with a misconfigured logger.
    '''
    global LOG
    if LOG is None:
        LOG = setup_logger()
    return LOG


CALLABLE = TypeVar('CALLABLE')

CALL_LEVEL = 0
INDENT_LEVEL = 0
INDENT_SPACES = '  ' * 1000
IS_TOP_LEVEL = True


def logged(**kwargs: Callable[[Any], Any]) -> Callable[[CALLABLE], CALLABLE]:
    '''
    Automatically wrap each invocation of the decorated function with logging it. Top-level calls
    are logged using the :py:const:`STEP` log level, with parameters logged at the :py:const:`PARAM`
    log level. Nested calls are logged at the ``DEBUG`` log level.

    By default parameters are logged by simply converting them to a string, with special cases for
    ``AnnData``, callable functions, boolean masks, vectors and matrices. You can override this by
    specifying ``parameter_name=convert_value_to_logged_value`` for the specific parameter.

    Expected usage is:

    .. code:: python

        @ut.logged()
        def some_function(...):
            ...
    '''
    formatter_by_name = kwargs

    def wrap(function: Callable) -> Callable:
        parameters = signature(function).parameters
        for name in formatter_by_name:
            if name not in parameters.keys():
                raise RuntimeError(f'formatter specified for the unknown parameter: {name} '
                                   f'for the function: {function.__module__}.{function.__qualname__}')
        ordered_parameters = list(parameters.values())

        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # pylint: disable=too-many-branches
            global CALL_LEVEL
            global INDENT_LEVEL
            names, values = \
                _collect_parameters(ordered_parameters, *args, **kwargs)
            adatas = _collect_adatas(values)
            new_adatas: List[AnnData] = []

            try:
                global IS_TOP_LEVEL
                old_is_top_level = IS_TOP_LEVEL

                for adata in adatas:
                    IS_TOP_LEVEL = False
                    if hasattr(adata, '__is_top_level__'):
                        IS_TOP_LEVEL = \
                            IS_TOP_LEVEL or getattr(adata, '__is_top_level__')
                    else:
                        IS_TOP_LEVEL = IS_TOP_LEVEL or CALL_LEVEL == 0
                        setattr(adata, '__is_top_level__', CALL_LEVEL == 0)
                        new_adatas.append(adata)

                if IS_TOP_LEVEL:
                    step_level = STEP
                    param_level = PARAM
                else:
                    step_level = DEBUG
                    param_level = DEBUG

                name = function.__qualname__
                if name[0] == '_':
                    name = name[1:]
                if logger().isEnabledFor(step_level):
                    logger().log(step_level, '%scall %s:',
                                 INDENT_SPACES[:2 * INDENT_LEVEL],
                                 name)
                    INDENT_LEVEL += 1
                CALL_LEVEL += 1

                if logger().isEnabledFor(param_level):
                    for name, value in zip(names, values):
                        log_value = _format_value(value, name,
                                                  formatter_by_name.get(name))
                        if log_value is not None:
                            logger().log(param_level, '%swith %s: %s',
                                         INDENT_SPACES[:2 * INDENT_LEVEL],
                                         name, log_value)

                return function(*args, **kwargs)

            finally:
                if logger().isEnabledFor(step_level):
                    INDENT_LEVEL -= 1
                CALL_LEVEL -= 1
                IS_TOP_LEVEL = old_is_top_level
                for adata in new_adatas:
                    delattr(adata, '__is_top_level__')

        return wrapper

    return wrap  # type: ignore


def _collect_parameters(
    parameters: List[Parameter],
    *args: Any,
    **kwargs: Any
) -> Tuple[List[str], List[Any]]:
    names: List[str] = []
    values: List[Any] = []

    for value, parameter in zip(args, parameters):
        names.append(parameter.name)
        values.append(value)

    for parameter in parameters[len(args):]:
        names.append(parameter.name)
        values.append(kwargs.get(parameter.name, parameter.default))

    return names, values


def _collect_adatas(values: List[Any]) -> List[AnnData]:
    adatas: List[AnnData] = []
    for value in values:
        if isinstance(value, AnnData):
            adatas.append(value)
        if isinstance(value, list):
            adatas += _collect_adatas(value)
    return adatas


def _format_value(  # pylint: disable=too-many-return-statements,too-many-branches
    value: Any,
    name: str,
    formatter: Optional[Callable[[Any], Any]] = None
) -> Optional[str]:
    if isinstance(value, Parameter.empty.__class__):
        return None

    checksum = ''
#   if utt.is_1d(value) and 'U' not in str(value.dtype) and 'o' not in str(value.dtype):
#       checksum = ' checksum: %.20e' % utt.shaped_checksum(value)
#   elif utt.is_2d(value):
#       checksum = ' checksum: %.20e' % utt.shaped_checksum(value)

    if formatter is not None:
        value = formatter(value)
        if value is None:
            return None

    if isinstance(value, AnnData):
        name = value.uns.get('__name__', 'unnamed')

        if hasattr(value.X, 'dtype'):
            dtype = getattr(value.X, 'dtype')
        else:
            dtype = 'unknown'

        return f'{name} annotated data with {value.shape[0]} X {value.shape[1]} {dtype}s' + checksum

    if isinstance(value, (np.bool_, bool, str, None.__class__)):
        return str(value) + checksum

    if isinstance(value, (int, float, np.float32, np.float64, np.int32, np.int64)):
        value = float(value)
        if 'random_seed' in name:
            if value == 0:
                return '0 (time)'
            return f'{value:g} (reproducible)'
        if 'rank' in name:
            return f'{value:g}'
        if 'fold' in name:
            return fold_description(value)
        if 'fraction' in name or 'quantile' in name:
            return fraction_description(value)
        if 'factor' in name:
            return f'X {value:.4g}'
        return f'{value:g}' + checksum

    if hasattr(value, '__qualname__'):
        return getattr(value, '__qualname__') + checksum

    if isinstance(value, (pd.Series, np.ndarray)) and value.ndim == 1 and value.dtype == 'bool':
        return mask_description(value) + checksum

    if hasattr(value, 'ndim'):
        if value.ndim == 2:
            text = f'{value.__class__.__name__} {value.shape[0]} X {value.shape[1]} {utt.matrix_dtype(value)}s'
            return text + checksum

        if value.ndim == 1:
            value = utt.to_numpy_vector(value)
            if len(value) > 100:
                text = f'{len(value)} {value.dtype}s'
                return text + checksum
            value = list(value)

    if isinstance(value, list):
        if len(value) > 100:
            return f'{len(value)} {value[0].__class__.__name__}s' + checksum
        texts = [element.uns.get('__name__', 'unnamed')
                 if isinstance(element, AnnData) else str(element)
                 for element in value]
        return f'[ {", ".join(texts)} ]' + checksum

    raise RuntimeError(  #
        f'unknown parameter type: {value.__class__} value: {value}')


def top_level(adata: AnnData) -> None:
    '''
    Indicate that the annotated data will be returned to the top-level caller, increasing its
    logging level.
    '''
    setattr(adata, '__is_top_level__', True)


def log_return(
    name: str,
    value: Any,
    *,
    formatter: Optional[Callable[[Any], Any]] = None
) -> bool:
    '''
    Log a ``value`` returned from a function with some ``name``.

    If ``formatter`` is specified, use it to override the default logged value formatting.
    '''
    if CALL_LEVEL == 0:
        top_log_level = INFO
    else:
        top_log_level = CALC

    return _log_value(name, value, 'return', None, top_log_level, formatter)


def logging_calc() -> bool:
    '''
    Whether we are actually logging the intermediate calculations.
    '''
    return (IS_TOP_LEVEL and logger().isEnabledFor(CALC)) \
        or (not IS_TOP_LEVEL and logger().isEnabledFor(DEBUG))


def log_calc(
    name: str,
    value: Any = None,
    *,
    formatter: Optional[Callable[[Any], Any]] = None
) -> bool:
    '''
    Log an intermediate calculated ``value`` computed from a function with some ``name``.

    If ``formatter`` is specified, use it to override the default logged value formatting.
    '''
    return _log_value(name, value, 'calc', None, CALC, formatter)


@contextmanager
def log_step(
    name: str,
    value: Any = None,
    *,
    formatter: Optional[Callable[[Any], Any]] = None
) -> Iterator[None]:
    '''
    Same as :py:func:`log_calc`, but also further indent all the log messages inside the ``with``
    statement body.
    '''
    global INDENT_LEVEL

    if log_calc(name, value, formatter=formatter):
        delta = 1
    else:
        delta = 0

    INDENT_LEVEL += delta
    try:
        yield
    finally:
        INDENT_LEVEL -= delta


#: Convert ``per`` to the name of the ``AnnData`` data member holding it.
MEMBER_OF_PER = \
    dict(m='uns',
         o='obs', v='var',
         oo='obsp', vv='varp',
         oa='obsm', va='varm',
         vo='layers')


def incremental(
    adata: AnnData,
    per: str,
    name: str,
    formatter: Optional[Callable[[Any], Any]] = None
) -> None:
    '''
    Declare that the named annotation will be built incrementally - set and then repeatedly modified.
    '''
    assert per in ('m', 'v', 'o', 'vv', 'oo', 'vo', 'va', 'oa')
    if not hasattr(adata, '__incremental__'):
        setattr(adata, '__incremental__', {})
    by_name: Dict[str, Tuple[str, Optional[Callable[[Any], Any]]]] = \
        getattr(adata, '__incremental__')
    assert name not in by_name
    by_name[name] = (per, formatter)


def done_incrementals(adata: AnnData) -> None:
    '''
    Declare that all the incremental values have been fully computed.
    '''
    assert hasattr(adata, '__incremental__')
    by_name: Dict[str, Tuple[str, Optional[Callable[[Any], Any]]]] = \
        getattr(adata, '__incremental__')
    setattr(adata, '__incremental__', [])

    for name, (per, formatter) in by_name.items():
        if name == '__x__':
            value = adata.X
        else:
            annotation = getattr(adata, MEMBER_OF_PER[per])
            if name not in annotation:
                raise RuntimeError(f'missing the incremental data: {name}')
            value = annotation[name]
        log_set(adata, per, name, value, formatter=formatter)


def cancel_incrementals(adata: AnnData) -> None:
    '''
    Cancel tracking incremental annotations.
    '''
    assert hasattr(adata, '__incremental__')
    delattr(adata, '__incremental__')


def log_set(
    adata: AnnData,
    per: str,
    name: str,
    value: Any,
    *,
    formatter: Optional[Callable[[Any], Any]] = None
) -> bool:
    '''
    Log setting some annotated data.
    '''
    assert per in ('m', 'v', 'o', 'vv', 'oo', 'vo', 'va', 'oa')

    is_top_level = \
        hasattr(adata, '__is_top_level__') \
        and getattr(adata, '__is_top_level__') \
        and (not hasattr(adata, '__incremental__')
             or not name in getattr(adata, '__incremental__'))

    adata_name = adata.uns.get('__name__', 'unnamed')
    if name == '__x__':
        name = f'{adata_name}.X'
    else:
        member_name = MEMBER_OF_PER[per]
        name = f'{adata_name}.{member_name}[{name}]'
    return _log_value(name, value, 'set', is_top_level, INFO, formatter)


def log_get(
    adata: AnnData,
    per: str,
    name: Any,
    value: Any,
    *,
    formatter: Optional[Callable[[Any], Any]] = None
) -> bool:
    '''
    Log getting some annotated data.
    '''
    assert per in ('m', 'v', 'o', 'vv', 'oo', 'vo', 'va', 'oa')

    is_top_level = \
        hasattr(adata, '__is_top_level__') \
        and getattr(adata, '__is_top_level__') \
        and (not hasattr(adata, '__incremental__')
             or not name in getattr(adata, '__incremental__'))

    adata_name = adata.uns.get('__name__', 'unnamed')
    if isinstance(name, str) and name == '__x__':
        name = f'{adata_name}.X'
    else:
        member_name = MEMBER_OF_PER[per]
        if not isinstance(name, str):
            name = '<data>'
        name = f'{adata_name}.{member_name}[{name}]'

    return _log_value(name, value, 'get', is_top_level, CALC, formatter)


def _log_value(
    name: str,
    value: Any,
    kind: str,
    is_top_level: Optional[bool],
    top_log_level: int,
    formatter: Optional[Callable[[Any], Any]] = None
) -> bool:
    if is_top_level is None:
        is_top_level = IS_TOP_LEVEL

    if is_top_level:
        level = top_log_level
    else:
        level = DEBUG

    if not logger().isEnabledFor(level):
        return False

    if value is None:
        logger().log(level, '%s%s', INDENT_SPACES[:2 * INDENT_LEVEL], name)
    else:
        log_value = _format_value(value, name, formatter)
        if name[0] == '-':
            logger().log(level, '%s%s: %s',
                         INDENT_SPACES[:2 * INDENT_LEVEL], name, log_value)
        else:
            logger().log(level, '%s%s %s: %s',
                         INDENT_SPACES[:2 * INDENT_LEVEL], kind, name, log_value)

    return True


def sizes_description(sizes: Union[utt.Vector, str]) -> str:
    '''
    Return a string for logging an array of sizes.
    '''
    if isinstance(sizes, str):
        return sizes

    sizes = utt.to_numpy_vector(sizes)
    mean = np.mean(sizes)
    return f'{len(sizes)} {sizes.dtype}s with mean {mean:.4g}'


def fractions_description(sizes: Union[utt.Vector, str]) -> str:
    '''
    Return a string for logging an array of fractions (between zero and one).
    '''
    if isinstance(sizes, str):
        return sizes

    sizes = utt.to_numpy_vector(sizes)
    mean = np.mean(sizes)
    percent = mean * 100
    return f'{len(sizes)} {sizes.dtype}s with mean {mean:.4g} ({percent:.4g}%)'


def groups_description(groups: Union[utt.Vector, str]) -> str:
    '''
    Return a string for logging an array of group indices.

    .. note::

        This assumes that the indices are consecutive, with negative values indicating "outliers".
    '''
    if isinstance(groups, str):
        return groups

    groups = utt.to_numpy_vector(groups)
    groups_count = np.max(groups) + 1
    outliers_count = np.sum(groups < 0)

    if groups_count == 0:
        assert outliers_count == len(groups)
        return f'{len(groups)} {groups.dtype} elements with all outliers (100%)'

    mean = (len(groups) - outliers_count) / groups_count
    return ratio_description(len(groups), f'{groups.dtype} element',
                             outliers_count, 'outliers') + \
        f' with {groups_count} groups with mean size {mean:.4g}'


def mask_description(mask: Union[str, utt.Vector]) -> str:
    '''
    Return a string for logging a boolean mask.
    '''
    if isinstance(mask, str):
        return mask

    mask = utt.to_numpy_vector(mask)
    if mask.dtype == 'bool':
        return ratio_description(mask.size, 'bool',
                                 np.sum(mask > 0), 'true')
    return ratio_description(mask.size, str(mask.dtype),
                             np.sum(mask > 0), 'positive')


def ratio_description(denominator: float, element: str, numerator: float, condition: str) -> str:
    '''
    Return a string for describing a ratio (including a percent representation).
    '''
    assert numerator >= 0
    assert denominator > 0

    if int(numerator) == numerator:
        numerator = int(numerator)
    if int(denominator) == denominator:
        denominator = int(denominator)

    percent = (numerator * 100) / denominator
    return f'{numerator} {condition} ({percent:.4g}%) out of {denominator} {element}s'


def progress_description(amount: int, index: int, element: str) -> str:
    '''
    Return a string for describing progress in a loop.
    '''
    assert amount > 0
    assert 0 <= index < amount

    percent = ((index + 1) * 100) / amount
    return f'#{index} out of {amount} {element}s ({percent:.4g}%)'


def fraction_description(fraction: Optional[float]) -> str:
    '''
    Return a string for describing a fraction (including a percent representation).
    '''
    if fraction is None:
        return 'None'
    percent = fraction * 100
    return f'{fraction:.4g} ({percent:.4g}%)'


def fold_description(fold: float) -> str:
    '''
    Return a string for describing a fraction (including a percent representation).
    '''
    if fold is None:
        return 'None'

    if fold >= 0:
        value = 2 ** fold
        char = 'X'
    else:
        value = 2 ** -fold
        char = '/'

    return f'{fold:.4g} ({char} {value:.4g})'
