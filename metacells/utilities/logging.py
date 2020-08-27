'''
Configure logging.

This provides a useful formatter which includes high-resolution time and thread names, and a set of
utility functions for effective logging of operations on annotation data.

The main tricky issue is picking the correct level for each log message. In general,
we want top-level operations, their key parameters, and writing of their results to
be ``INFO`` - and everything else to be ``DEBUG``.

To achieve this, we track for each ``AnnData`` whether it is a temporary object, and use the
:py:func:`log_level` function below. The code carefully uses this level for the potentially
top-level informative messages, and uses ``DEBUG`` for everything else.

We also allow each ``AnnData`` object to have an optional name for logging. Both this name and
whether the data is temporary are specified when we :py:func:`metacells.utilities.annotation.setup`
or :py:func:`metacells.utilities.annotation.slice` some data.
'''

import logging
import sys
from datetime import datetime
from logging import Formatter, Logger, StreamHandler, getLogger
from typing import Any, Optional, Tuple

import numpy as np  # type: ignore
from anndata import AnnData

import metacells.utilities.documentation as utd
import metacells.utilities.typing as utt

__all__ = [
    'setup_logger',
    'log_level',
    'log_operation',
    'log_of',
    'log_mask',
    'mask_description',
    'ratio_description',
    'fraction_description',
]


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
        return '%s.%03d' % (seconds, record.msecs)


@utd.expand_doc()
def setup_logger(*, level: str = 'WARN', prefix: Optional[str] = None) -> Logger:
    '''
    Setup logging to stderr at some ``level`` (default: {level}).

    If ``prefix`` (default: {prefix}) is specified, it is added to each logged message.
    '''
    if prefix is None:
        log_format = '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
    else:
        log_format = '%(asctime)s - ' + prefix \
            + ' - %(threadName)s - %(levelname)s - %(message)s'
    handler = StreamHandler(sys.stderr)
    handler.setFormatter(LoggingFormatter(log_format))
    logger = getLogger('metacells')
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_level(adata: AnnData) -> int:
    '''
    Return the log level for operations on the data.

    By default, this is ``INFO``. For ``tmp`` data, this is ``DEBUG``.
    '''
    return logging.DEBUG if adata.uns.get('__tmp__', False) else logging.INFO


def log_operation(
    logger: Logger,
    adata: AnnData,
    operation: str,
    of: Optional[str] = '__no_of__',
    default: Optional[str] = None,
) -> Tuple[str, int]:
    '''
    Log the start of some ``operation``.

    Also log the ``of`` data it applies to, similarly to using :py:func:`log_of`. If the (default)
    special value ``__no_of__`` is given, then the operation has no ``of`` data (e.g.,
    py:func:`metacells.tools.named_genes.find_named_genes`).

    Return the explicit ``of`` data name and the log level for the ``adata``.
    '''
    level = log_level(adata)
    name = adata.uns.get('__name__')
    of = of or default or adata.uns['__focus__']

    texts = [operation]
    if of != '__no_of__':
        texts.append(of)
    if name is not None:
        texts.append(name)

    logger.log(level, '%s ...', ' of '.join(texts))

    return of, level


def log_of(
    logger: Logger,
    adata: AnnData,
    of: Optional[str],
    default: Optional[str] = None,
    *,
    name: str = 'of'
) -> str:
    '''
    Given an ``of`` parameter for an operation, compute and return the actual value
    using the ``default`` or the focus, and log it properly using the ``name``.
    '''
    if of is None:
        of = default or adata.uns['__focus__']
        level = logging.DEBUG
    else:
        level = log_level(adata)

    logger.log(level, '  %s: %s', name, of)
    return of


def log_mask(
    logger: Logger,
    level: int,
    name: str,
    mask: utt.Vector
) -> None:
    '''
    Log the computation of a boolean mask.
    '''
    if logger.isEnabledFor(level):
        logger.log(level, '  %s: %s', name, mask_description(mask))


def mask_description(mask: utt.Vector) -> str:
    '''
    Return a string for logging a boolean mask.

    This returns the number of set entries, the total number of entries, and the percentage.
    '''
    mask = utt.to_dense_vector(mask)
    return 'mask of ' + ratio_description(np.sum(mask), mask.size)


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

    return '%s / %s (%.2f%%)' % (numerator, denominator, numerator * 100 / denominator)


def fraction_description(fraction: float) -> str:
    '''
    Return a string for describing a fraction (including a percent representation).
    '''
    return '%s (%.2f%%)' % (fraction, fraction * 100)
