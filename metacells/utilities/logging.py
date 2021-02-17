'''
Logging
-------

This provides a useful formatter which includes high-resolution time and thread names, and a set of
utility functions for effective logging of operations on annotation data.

The main tricky issue is picking the correct level for each log message. In general,
we want top-level operations, their key parameters, and writing of their results to
be ``INFO`` - and everything else to be ``DEBUG``.

To achieve this, we track for each ``AnnData`` whether it is a temporary object, and use the
``get_log_level`` function below. The code carefully uses this level for the potentially
top-level informative messages, and uses ``DEBUG`` for everything else.

We also allow each ``AnnData`` object to have an optional name for logging. Both this name and
whether the data is temporary are specified when we ``setup`` or
:py:func:`metacells.utilities.annotation.slice` some data.
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
    'groups_description',
    'mask_description',
    'ratio_description',
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
        msecs = round(record.msecs * 1000)
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
    level: int = logging.WARN,
    to: IO = sys.stderr,
    time: bool = True,
    process: Optional[bool] = None,
    name: Optional[str] = None,
    short_level_names: bool = True,
) -> Logger:
    '''
    Setup the global :py:func:`logger`.

    .. note::

        A second call will fail as the logger will already be set up.

    Logging from multiple sub-processes (e.g., using (e.g., using
    :py:func:`metacells.utilities.parallel.parallel_map`) will synchronize using a global lock so
    messages will not get garbled.

    If ``level`` is not specified, only warnings and errors will be logged.

    If ``to`` is not specified, the output is sent to ``sys.stderr``.

    If ``time`` (default: {time}), include a millisecond-resolution timestamp for each message.

    If ``name`` (default: {name}) is specified, it is added to each logged message.

    If ``process`` (default: {process}), include the process index (the main process has the index
    zero). If ``None``, then is set if :py:func:`metacells.utilities.parallel.get_processors_count`
    is greater than one (that is, multiprocessing is used).

    If ``short_level_names``, the log level names are shortened to three characters, for consistent
    formatting of indented (nested) log messages.

    .. note::

        Invoking :py:func:`setup_logger` replaces the name of the main thread to ``#0`` to make it
        more compatible with the sub-thread names.
    '''
    global LOG
    assert LOG is None

    log_format = '%(levelname)s - %(message)s'

    current_thread().name = '#0'

    if process is None:
        process = utp.get_processors_count() > 1

    if process:
        log_format = '%(threadName)s - ' + log_format

    if name is not None:
        log_format = name + ' - ' + log_format
    if time:
        log_format = '%(asctime)s - ' + log_format

    handler = StreamHandler(to)
    if short_level_names:
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

NESTING_LEVEL = 0
NESTING_INDENT = '  ' * 1000
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
            global NESTING_LEVEL
            names, values = \
                _collect_parameters(ordered_parameters, *args, **kwargs)
            adatas = [value for value in values if isinstance(value, AnnData)]
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
                        IS_TOP_LEVEL = IS_TOP_LEVEL or NESTING_LEVEL == 0
                        setattr(adata, '__is_top_level__', NESTING_LEVEL == 0)
                        new_adatas.append(adata)

                if IS_TOP_LEVEL:
                    step_level = STEP
                    param_level = PARAM
                else:
                    step_level = DEBUG
                    param_level = DEBUG

                if logger().isEnabledFor(step_level):
                    logger().log(step_level, '%scall %s:',
                                 NESTING_INDENT[:2 * NESTING_LEVEL],
                                 function.__qualname__)
                    NESTING_LEVEL += 1

                if logger().isEnabledFor(param_level):
                    for name, value in zip(names, values):
                        log_value = _format_value(value, name,
                                                  formatter_by_name.get(name))
                        if log_value is not None:
                            logger().log(param_level, '%swith %s: %s',
                                         NESTING_INDENT[:2 * NESTING_LEVEL],
                                         name, log_value)

                return function(*args, **kwargs)

            finally:
                if logger().isEnabledFor(step_level):
                    NESTING_LEVEL -= 1
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


def _format_value(  # pylint: disable=too-many-return-statements,too-many-branches
    value: Any,
    name: str,
    formatter: Optional[Callable[[Any], Any]] = None
) -> Optional[str]:
    if isinstance(value, Parameter.empty.__class__):
        return None

    if formatter is not None:
        value = formatter(value)
        if value is None:
            return None

    if isinstance(value, AnnData):
        if hasattr(value, '__name__'):
            name = getattr(value, '__name__') + ' -'
        else:
            name = 'adata'

        if hasattr(value.X, 'dtype'):
            dtype = getattr(value.X, 'dtype')
        else:
            dtype = 'unknown'

        return f'{name} annotated data of {value.shape} {dtype}s'

    if isinstance(value, (str, None.__class__)):
        return str(value)

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
        return f'{value:g}'

    if hasattr(value, '__qualname__'):
        return getattr(value, '__qualname__')

    if isinstance(value, (pd.Series, np.ndarray)) and value.ndim == 1 and value.dtype == 'bool':
        return f'boolean mask of {mask_description(value)}'

    if hasattr(value, 'ndim'):
        if value.ndim == 2:
            return f'a matrix of {value.shape} {value.dtype}s'
        if value.ndim == 1:
            if 'U' not in str(value.dtype):
                return f'a vector of {len(value)} {value.dtype}s'
            value = list(value)

    if isinstance(value, list):
        return f'[ {", ".join([str(element) for element in value])} ]'

    raise RuntimeError(  #
        f'unknown parameter type: {value.__class__} value: {value}')


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
    return _log_value(name, value, 'return', None, INFO, formatter)


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
    global NESTING_LEVEL

    if log_calc(name, value, formatter=formatter):
        delta = 1
    else:
        delta = 0

    NESTING_LEVEL += delta
    try:
        yield
    finally:
        NESTING_LEVEL -= delta


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

    adata_name = adata.uns.get('__name__', 'adata')
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

    adata_name = adata.uns.get('__name__', 'adata')
    if name == '__x__':
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
    top_level: int,
    formatter: Optional[Callable[[Any], Any]] = None
) -> bool:
    if is_top_level is None:
        is_top_level = IS_TOP_LEVEL

    if is_top_level:
        level = top_level
    else:
        level = DEBUG

    if not logger().isEnabledFor(level):
        return False

    if value is None:
        logger().log(level, '%s%s', NESTING_INDENT[:2 * NESTING_LEVEL], name)
    else:
        log_value = _format_value(value, name, formatter)
        if name[0] == '-':
            logger().log(level, '%s%s: %s',
                         NESTING_INDENT[:2 * NESTING_LEVEL], name, log_value)
        else:
            logger().log(level, '%s%s %s: %s',
                         NESTING_INDENT[:2 * NESTING_LEVEL], kind, name, log_value)

    return True


def sizes_description(sizes: Union[utt.Vector, str]) -> str:
    '''
    Return a string for logging an array of sizes.

    This returns the mean size.
    '''
    if isinstance(sizes, str):
        return sizes

    sizes = utt.to_numpy_vector(sizes)
    mean = np.mean(sizes)
    return f'vector of {len(sizes)} {sizes.dtype} sizes with mean {mean:.4g}'


def groups_description(groups: Union[utt.Vector, str]) -> str:
    '''
    Return a string for logging an array of group indices.

    This returns the number and mean size of the groups.
    '''
    if isinstance(groups, str):
        return groups

    groups = utt.to_numpy_vector(groups)
    text = f'vector of {len(groups)} {groups.dtype} groups'
    groups_count = np.max(groups) + 1
    outliers_count = np.sum(groups < 0)
    if groups_count > 0:
        mean = (len(groups) - outliers_count) / groups_count
        text += f' with {groups_count} distinct groups with mean size {mean:.4g}'
    if outliers_count > 0:
        outliers = ratio_description(outliers_count, groups.size)
        text += f' and {outliers} outliers'
    return text


def mask_description(mask: Union[str, utt.Vector]) -> str:
    '''
    Return a string for logging a boolean mask.

    This returns the number of set entries, the total number of entries, and the percentage.
    '''
    if isinstance(mask, str):
        return mask

    mask = utt.to_numpy_vector(mask)
    return ratio_description(np.sum(mask > 0), mask.size)


def ratio_description(numerator: float, denominator: float) -> str:
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
    return f'{numerator} / {denominator} ({percent:.4g}%)'


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
