'''
Wrappers for C++ extension functions.
'''

from typing import Optional

import numpy as np  # type: ignore

import metacells.extensions as xt  # type: ignore

__all__ = [
    'downsample',
    'downsample_tmp_size',
]

DATA_TYPES = ['float32', 'float64', 'int32', 'int64', 'uint32', 'uint64']


def downsample(
    data: np.ndarray,
    samples: int,
    *,
    tmp: Optional[np.ndarray] = None,
    output: Optional[np.ndarray] = None,
    random_seed: int = 0
) -> None:
    '''
    Downsample a vector of sample counters.

    **Input**

    * A Numpy array ``data`` containing non-negative integer sample counts.

    * A desired total number of ``samples``.

    * An optional temporary storage Numpy array minimize allocations for multiple invocations.

    * An optional Numpy array ``output`` to hold the results (otherwise, the input is overwritten).

    * An optional random seed to make the operation replicable.

    The arrays may have any of the data types: ``float32``, ``float64``, ``int32``, ``int64``,
    ``uint32``, ``uint64``.

    **Operation**

    If the total number of samples (sum of the data array) is not higher than the required number of
    samples, the output is identical to the input.

    Otherwise, treat the input as if it was a set where each index appeared its data count number of
    times. Randomly select the desired number of samples from this set (without repetition), and
    store in the output the number of times each index was chosen.

    **Motivation**

    Downsampling is an effective way to get the same number of samples in multiple observations (in
    particular, the same number of total UMIs in multiple cells), and serves as an alternative to
    normalization (e.g., working with UMI fractions instead of raw UMI counts).

    Downsampling is especially important when computing correlations between observations. When
    there is high variance between the total samples count in different observations (total UMI
    count in different cells), then normalization will return higher values when correlating
    observations with a higher sample count, which will result in an inflated estimation of their
    similarity to other observations. Downsampling avoids this effect.
    '''
    if tmp is None:
        tmp = np.empty(downsample_tmp_size(data.size), dtype='int32')
    else:
        tmp.resize(downsample_tmp_size(data.size))

    if output is None:
        output = data

    function_name = \
        'downsample_%s_%s_%s' % (data.dtype, tmp.dtype, output.dtype)
    function = getattr(xt, function_name)
    function(data, tmp, output, samples, random_seed)


def downsample_tmp_size(size: int) -> int:
    '''
    Return the size of the temporary array needed to ``downsample`` data of the specified size.
    '''
    return xt.downsample_tmp_size(size)
