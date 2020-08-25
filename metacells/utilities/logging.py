'''
Configure logging.
'''

import sys
from datetime import datetime
from logging import Formatter, Logger, StreamHandler, getLogger
from typing import Any, Optional

import metacells.utilities.documentation as utd

__all__ = [
    'setup_logger',
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
