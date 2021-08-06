'''
Annotation
----------

In general we are using ``AnnData`` to hold the data being analyzed. However, the interface
of AnnData leaves some things out which are crucial for the proper working of our algorithm
(and any other algorithm that works at a scale of millions of cells).

X as an Annotation
..................

For a uniform interface, we pretend the ``X`` member is a per-variable-per-observation annotation
with the special name ``__x__``. This allows us to have APIs that take an annotation name and just
pass them (typically by default) the annotation "name" ``__x__`` to force the code to run on the
``X`` data member.

In general the APIs allow specifying either annotation names or alternatively an explicit matrix (or
vector for per-observation or per-variable annotations), for maximal usage flexibility.

Data Types
..........

The generic ``AnnData`` is cheerfully permissive when it comes to the data it contains. That is, when
accessing data, it isn't clear whether you'll be getting a numpy array or a pandas data series, and
for 2D data you might be getting all sort of data types (including sparse matrices of various
formats).

Python itself is very loose about the interface these data types provide - some operations such as
``len`` and ``shape`` and accessing an element by integer indices are safe, more advanced operations
can silently produce the wrong results, and most operations work on a subset of the data types,
often with wildly incompatible interfaces.

To combat this, we have the :py:mod:`metacells.utilities.typing` module which imposes some order on
the types zoo, and, in addition, we provide here accessor functions which return deterministic
usable data types, allowing for safe processing of the results. This is combined with the
py:mod:`metacells.utilities.computation` module which provides a set of operations that work
consistently on the few data types we use.

Data Layout
...........

A related issue is the layout of 2D data. For small matrices, this doesn't matter, but when dealing
with large matrices (millions of rows/columns), performing a simple operation may takes orders of
magnitude longer if applied to a matrix of the wrong layout.

To make things worse the builtin functions for converting between matrix layouts are pretty
inefficient so more efficient variants are provided in the py:mod:`metacells.utilities.computation`
module.

The accessors in this module allow for explicitly controlling the layout of the data they return,
and cache the different layouts of the same annotations of the ``AnnData`` (under the reasonable
assumption that the original data is not modified). This allows for writing
guaranteed-to-be-efficient processing code.

Data Logging
............

A side benefit of exclusively using the accessors provided here is that they participate in the
automated logging provided by the :py:mod:`metacells.utilities.logging` module. That is, using them
will automatically log writing the final results of a computation to the user at the ``INFO`` log
level, while higher logging level give insight into the exact data being read and written by the
algorithm's nested sub-steps.
'''

from typing import (Any, Callable, Collection, Dict, List, MutableMapping,
                    Optional, Tuple, Union)

import anndata as ad
import numpy as np
from anndata import AnnData

import metacells.utilities.computation as utc
import metacells.utilities.logging as utl
import metacells.utilities.timing as utm
import metacells.utilities.typing as utt

__all__ = [
    'slice',
    'copy_adata',

    'set_name',
    'get_name',

    'set_m_data',
    'get_m_data',

    'set_o_data',
    'get_o_series',
    'get_o_numpy',
    'get_o_names',
    'maybe_o_numpy',

    'set_v_data',
    'get_v_series',
    'get_v_numpy',
    'get_v_names',
    'maybe_v_numpy',

    'set_oo_data',
    'get_oo_frame',
    'get_oo_proper',

    'set_vv_data',
    'get_vv_frame',
    'get_vv_proper',

    'set_oa_data',
    'get_oa_frame',
    'get_oa_proper',

    'set_va_data',
    'get_va_frame',
    'get_va_proper',

    'set_vo_data',
    'get_vo_frame',
    'get_vo_proper',

    'has_data',
]


Annotations = Union[MutableMapping[Any, Any], utt.PandasFrame]


@utm.timed_call()
def slice(  # pylint: disable=redefined-builtin,too-many-branches,too-many-statements
    adata: AnnData,
    *,
    name: Optional[str] = None,
    obs: utt.Vector = None,
    vars: utt.Vector = None,
    track_obs: Optional[str] = None,
    track_var: Optional[str] = None,
    share_derived: bool = True,
    top_level: bool = True,
) -> AnnData:
    '''
    Return new annotated data which includes a subset of the full ``adata``.

    If ``name`` is not specified, the data will be unnamed. Otherwise, if it starts with a ``.``, it
    will be appended to the current name (if any). Otherwise, ``name`` is the new name.

    If ``obs`` and/or ``vars`` are specified, they should be set to either a boolean mask or a
    collection of indices to include in the data slice. In the case of an indices array, it is
    assumed the indices are unique and sorted, that is that their effect is similar to a mask.

    If ``track_obs`` and/or ``track_var`` are specified, the result slice will include a
    per-observation and/or per-variable annotation containing the indices of the sliced elements in
    the original full data.

    If the slice happens to be the full original data, then this becomes equivalent to
    :py:func:`copy_adata`, and by default this will ``share_derived`` (share the derived data
    cache).
    '''
    assert '__x__' not in adata.layers

    is_same_obs: Optional[bool] = None
    if obs is None:
        obs = range(adata.n_obs)
        is_same_obs = True
    else:
        assert 0 < len(obs) <= adata.n_obs
        if len(obs) < adata.n_obs:
            is_same_obs = False

        obs = utt.to_numpy_vector(obs)
        if obs.dtype == 'bool':
            assert obs.size == adata.n_obs
            assert np.any(obs)
            is_same_obs = bool(np.all(obs))

    is_same_vars: Optional[bool] = None
    if vars is None:
        vars = range(adata.n_vars)
        is_same_vars = True
    else:
        assert 0 < len(vars) <= adata.n_vars
        if len(vars) < adata.n_vars:
            is_same_vars = False

        vars = utt.to_numpy_vector(vars)
        if vars.dtype == 'bool':
            assert vars.size == adata.n_vars
            assert np.any(vars)
            is_same_vars = bool(np.all(vars))

    if is_same_obs and is_same_vars:
        bdata = copy_adata(adata, name=name, share_derived=share_derived,
                           top_level=top_level)

    else:
        if is_same_vars:
            replaced = _replace_with_layout(adata, 'row_major')
        elif is_same_obs:
            replaced = _replace_with_layout(adata, 'column_major')
        else:
            replaced = {}

        try:
            with utm.timed_step('adata.slice'):
                bdata = adata[obs, vars].copy()
        finally:
            _replace_back(adata, replaced)

        set_name(bdata, name)

        if hasattr(bdata, '__is_top_level__'):
            delattr(bdata, '__is_top_level__')
        if hasattr(bdata, '__derived__'):
            delattr(bdata, '__derived__')

        if share_derived \
                and (is_same_obs or is_same_obs is None) \
                and (is_same_vars or is_same_vars is None):

            if is_same_obs is None:
                is_same_obs = bdata.n_obs == adata.n_obs \
                    and bool(np.all(bdata.obs_names == adata.obs_names))

            if is_same_vars is None:
                is_same_vars = bdata.n_vars == adata.n_vars \
                    and bool(np.all(bdata.var_names == adata.var_names))

            if is_same_obs and is_same_vars:
                if not hasattr(adata, '__derived__'):
                    setattr(adata, '__derived__', {})
                setattr(bdata, '__derived__', getattr(adata, '__derived__'))

        if top_level:
            utl.top_level(bdata)

    if utl.logging_calc():
        utl.log_calc(  #
            f'slice {get_name(adata, "unnamed")} into {get_name(bdata, "unnamed")} shape {bdata.shape}')

    if track_obs is not None:
        set_o_data(bdata, track_obs, np.arange(adata.n_obs)[obs])

    if track_var is not None:
        set_v_data(bdata, track_var, np.arange(adata.n_vars)[vars])

    return bdata


@utm.timed_call()
def _replace_with_layout(adata: AnnData, layout: str) -> Dict[str, utt.Matrix]:
    replaced: Dict[str, utt.Matrix] = {}

    matrix: utt.Matrix = adata.X  # type: ignore
    if not utt.is_layout(matrix, layout):
        replaced['__x__'] = matrix
        adata.X = get_vo_proper(adata, '__x__', layout=layout)

    for name in adata.layers:
        matrix = adata.layers[name]
        if not utt.is_layout(matrix, layout):
            replaced[name] = matrix
            adata.layers[name] = get_vo_proper(adata, name, layout=layout)

    return replaced


@utm.timed_call()
def _replace_back(adata: AnnData, replaced: Dict[str, utt.Matrix]) -> None:
    for name, matrix in replaced.items():
        if name == '__x__':
            adata.X = matrix
        else:
            adata.layers[name] = matrix


@utm.timed_call()
def copy_adata(
    adata: AnnData,
    *,
    name: Optional[str] = None,
    share_derived: bool = True,
    top_level: bool = True
) -> AnnData:
    '''
    Return a copy of some annotated ``adata``.

    If ``name`` is not specified, the data will be unnamed. Otherwise, if it starts with a ``.``, it
    will be appended to the current name (if any). Otherwise, ``name`` is the new name.

    If ``share_derived`` is ``True`` (the default), then the copy will share the derived data cache,
    which contains specific layout variants of matrix data and sums of columns/rows of matrix data.
    Use this if you intend to modify the copy in-place.

    .. note::

        In general we assume annotated data is **not** modified in-place, but it might make sense to
        create a copy (**not** sharing derived data), modify it immediately (before accessing data
        in a specific layout), and then proceed to process it without further modifications.
    '''
    with utm.timed_step('adata.copy'):
        bdata = adata.copy()

    set_name(bdata, name)

    if hasattr(bdata, '__is_top_level__'):
        delattr(bdata, '__is_top_level__')

    if share_derived:
        if not hasattr(adata, '__derived__'):
            setattr(adata, '__derived__', {})
        setattr(bdata, '__derived__', getattr(adata, '__derived__'))
    else:
        if hasattr(bdata, '__derived__'):
            delattr(bdata, '__derived__')

    if top_level:
        utl.top_level(bdata)
    return bdata


def has_data(
    adata: AnnData,
    name: str,
    layout: Optional[str] = None,
) -> bool:
    '''
    Test whether we have the specified data.

    If the data is per-variable-per-observation, and ``layout`` is specified (one of ``row_major``
    and ``column_major``), it returns whether the specific data layout is available in the cache,
    without having to re-layout existing data.
    '''
    assert layout is None or layout in utt.LAYOUT_OF_AXIS

    derived: Union[Dict[str, Any], List[str]] = []
    if hasattr(adata, '__derived__'):
        derived = getattr(adata, '__derived__')

    if name == '__x__':
        if layout is None:
            return True
        matrix: utt.Matrix = adata.X  # type: ignore
        return utt.matrix_layout(matrix) == layout \
            or f'vo:__x__:{layout}' in derived

    for per, annotations in (('vo', adata.layers),
                             ('oo', adata.obsp),
                             ('vv', adata.varp),
                             ('va', adata.obsm),
                             ('oa', adata.varm)):
        if name not in annotations:
            continue
        if layout is None:
            return True
        matrix = annotations[name]
        return utt.matrix_layout(matrix) == layout \
            or f'{per}:{name}:{layout}' in derived

    assert layout is None

    for annotations in (adata.obs, adata.var, adata.uns):
        if name in annotations:
            return True

    return False


def get_m_data(
    adata: AnnData,
    name: str,
    *,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> Any:
    '''
    Get metadata (unstructured annotation) in ``adata`` by its ``name``.
    '''
    data = adata.uns.get(name)
    if data is None:
        raise _unknown_data(adata, name, 'm')
    utl.log_get(adata, 'm', name, data, formatter=formatter)
    return data


def _get_o_data(
    adata: AnnData,
    name: Union[str, utt.Shaped],
    *,
    sum: bool,  # pylint: disable=redefined-builtin
    formatter: Optional[Callable[[Any], Any]] = None,
) -> Any:
    if not isinstance(name, str):
        log_name = '<data>'
    else:
        if name.endswith('|sum'):
            name = name[:-4]
            assert not sum
            sum = True
        log_name = name

    if sum:
        if formatter is None:
            formatter = utl.sizes_description
        data = _get_vo_sum_data(adata, 'row', name)
        log_name += '|sum'
    else:
        data = _get_shaped_data(adata, 'o', adata.obs,
                                shape=(adata.n_obs,), name=name)
    utl.log_get(adata, 'o', log_name, data, formatter=formatter)
    return data


def get_o_series(
    adata: AnnData,
    name: Union[str, utt.Shaped],
    *,
    sum: bool = False,  # pylint: disable=redefined-builtin
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.PandasSeries:
    '''
    Get per-observation (cell) data in ``adata`` by its ``name`` as a pandas series.

    If ``name`` is a string, it is the name of a per-observation annotation to fetch.
    Otherwise, it should be some vector of data of the appropriate size.
    '''
    data = _get_o_data(adata, name, sum=sum, formatter=formatter)
    series = utt.maybe_pandas_series(data)
    if series is None:
        series = utt.to_pandas_series(data, index=adata.obs_names)
    return series


def get_o_numpy(
    adata: AnnData,
    name: Union[str, utt.Shaped],
    *,
    sum: bool = False,  # pylint: disable=redefined-builtin
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.NumpyVector:
    '''
    Get per-observation (cell) data in ``adata`` by its ``name`` as a Numpy array.

    If ``name`` is a string, it is the name of a per-observation annotation to fetch.
    Otherwise, it should be some vector of data of the appropriate size.

    If ``sum`` is ``True``, then ``name`` should be the name of a per-observation-per-variable
    annotation, or a matrix, and this will return the sum (per row) of this data.
    '''
    data = _get_o_data(adata, name, sum=sum, formatter=formatter)
    return utt.to_numpy_vector(data)


def get_o_names(
    adata: AnnData,
) -> utt.NumpyVector:
    '''
    Get a numpy vector of observation names.
    '''
    data = utt.to_numpy_vector(adata.obs_names)
    utl.log_get(adata, 'o', '__name__', data)
    return data


def maybe_o_numpy(
    adata: AnnData,
    name: Union[str, utt.Shaped, None],
    *,
    sum: bool = False,  # pylint: disable=redefined-builtin
    formatter: Optional[Callable[[Any], Any]] = None,
) -> Optional[utt.NumpyVector]:
    '''
    Similar to :py:func:`get_o_numpy`, but if ``name`` is ``None``, return ``None``.
    '''
    if name is None:
        return None
    return get_o_numpy(adata, name, sum=sum, formatter=formatter)


def _get_v_data(
    adata: AnnData,
    name: Union[str, utt.Shaped],
    *,
    sum: bool,  # pylint: disable=redefined-builtin
    formatter: Optional[Callable[[Any], Any]] = None,
) -> Any:
    if not isinstance(name, str):
        log_name = '<data>'
    else:
        if isinstance(name, str) and name.endswith('|sum'):
            name = name[:-4]
            assert not sum
            sum = True
        log_name = name

    if sum:
        if formatter is None:
            formatter = utl.sizes_description
        data = _get_vo_sum_data(adata, 'column', name)
        log_name += '|sum'
    else:
        data = _get_shaped_data(adata, 'v', adata.var,
                                shape=(adata.n_vars,), name=name)
    utl.log_get(adata, 'v', log_name, data, formatter=formatter)
    return data


def get_v_series(
    adata: AnnData,
    name: Union[str, utt.Shaped],
    *,
    sum: bool = False,  # pylint: disable=redefined-builtin
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.PandasSeries:
    '''
    Get per-variable (gene) data in ``adata`` by its ``name`` as a pandas series.

    If ``name`` is a string, it is the name of a per-variable annotation to fetch.
    Otherwise, it should be some vector of data of the appropriate size.
    '''
    data = _get_v_data(adata, name, sum=sum, formatter=formatter)
    series = utt.maybe_pandas_series(data)
    if series is None:
        series = utt.to_pandas_series(data, index=adata.var_names)
    return series


def get_v_numpy(
    adata: AnnData,
    name: Union[str, utt.Shaped],
    *,
    sum: bool = False,  # pylint: disable=redefined-builtin
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.NumpyVector:
    '''
    Get per-variable (gene) data in ``adata`` by its ``name`` as a numpy array.

    If ``name`` is a string, it is the name of a per-variable annotation to fetch.
    Otherwise, it should be some vector of data of the appropriate size.
    '''
    data = _get_v_data(adata, name, sum=sum, formatter=formatter)
    return utt.to_numpy_vector(data)


def get_v_names(
    adata: AnnData,
) -> utt.NumpyVector:
    '''
    Get a numpy vector of variable names.
    '''
    data = utt.to_numpy_vector(adata.var_names)
    utl.log_get(adata, 'v', '__name__', data)
    return data


def maybe_v_numpy(
    adata: AnnData,
    name: Union[str, utt.Shaped, None],
    *,
    sum: bool = False,  # pylint: disable=redefined-builtin
    formatter: Optional[Callable[[Any], Any]] = None,
) -> Optional[utt.NumpyVector]:
    '''
    Similar to :py:func:`get_v_numpy`, but if ``name`` is ``None``, return ``None``.
    '''
    if name is None:
        return None
    return get_v_numpy(adata, name, sum=sum, formatter=formatter)


def _get_oo_data(
    adata: AnnData,
    name: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.Matrix:
    data = _get_layout_data(adata, 'oo', adata.obsp,
                            shape=(adata.n_obs, adata.n_obs),
                            name=name, layout=layout)
    utl.log_get(adata, 'oo', name, data, formatter=formatter)
    return data


def get_oo_frame(
    adata: AnnData,
    name: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.PandasFrame:
    '''
    Get per-observation-per-observation (per-cell-per-cell) data as a pandas data frame.

    If ``name`` is a string, it is the name of a per-variable annotation to fetch.
    Otherwise, it should be some matrix of data of the appropriate size.

    If ``layout`` (default: {layout}) is specified, it must be one of ``row_major`` or
    ``column_major``. If this requires relayout of the data, the result is cached in a hidden data
    member for future reuse.
    '''
    data = _get_oo_data(adata, name, layout=layout, formatter=formatter)
    frame = utt.maybe_pandas_frame(data)
    if frame is None:
        frame = utt.to_pandas_frame(utt.to_proper_matrix(data),
                                    index=adata.obs_names,
                                    columns=adata.obs_names)
    return frame


def get_oo_proper(
    adata: AnnData,
    name: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.ProperMatrix:
    '''
    Same as ``get_oo_data`` but returns a :py:const:`metacells.utilities.typing.ProperMatrix`.
    '''
    data = _get_oo_data(adata, name, layout=layout, formatter=formatter)
    return utt.to_proper_matrix(data, default_layout=layout or 'row_major')


def _get_vv_data(
    adata: AnnData,
    name: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.Matrix:
    data = _get_layout_data(adata, 'vv', adata.varp,
                            shape=(adata.n_vars, adata.n_vars),
                            name=name, layout=layout)
    utl.log_get(adata, 'vv', name, data, formatter=formatter)
    return data


def get_vv_frame(
    adata: AnnData,
    name: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.PandasFrame:
    '''
    Get per-variable-per-variable (per-gene-per-gene) data as a pandas data frame.

    If ``name`` is a string, it is the name of a per-variable annotation to fetch.
    Otherwise, it should be some matrix of data of the appropriate size.

    If ``layout`` (default: {layout}) is specified, it must be one of ``row_major`` or
    ``column_major``. If this requires relayout of the data, the result is cached in a hidden data
    member for future reuse.
    '''
    data = _get_vv_data(adata, name, layout=layout, formatter=formatter)
    frame = utt.maybe_pandas_frame(data)
    if frame is None:
        frame = utt.to_pandas_frame(utt.to_proper_matrix(data),
                                    index=adata.var_names,
                                    columns=adata.var_names)
    return frame


def get_vv_proper(
    adata: AnnData,
    name: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.ProperMatrix:
    '''
    Same as ``get_vv_data`` but returns a :py:const:`metacells.utilities.typing.ProperMatrix`.
    '''
    data = _get_vv_data(adata, name, layout=layout, formatter=formatter)
    return utt.to_proper_matrix(data, default_layout=layout or 'row_major')


def _get_oa_data(
    adata: AnnData,
    name: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.Matrix:
    data = _get_layout_data(adata, 'oa', adata.obsm,
                            shape=(adata.n_obs, 0),
                            name=name, layout=layout)
    utl.log_get(adata, 'oa', name, data, formatter=formatter)
    return data


def get_oa_frame(
    adata: AnnData,
    name: Union[str, utt.Matrix],
    *,
    columns: Optional[Collection],
    layout: Optional[str] = None,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.PandasFrame:
    '''
    Get per-observation-per-any (per-cell-per-any) data as a pandas data frame.

    Rows are observations (cells), indexed by the observation names (typically cell barcode).
    Columns are "something" - specify ``columns`` to specify an index.

    If ``name`` is a string, it is the name of a per-variable annotation to fetch.
    Otherwise, it should be some matrix of data of the appropriate size.

    If ``layout`` (default: {layout}) is specified, it must be one of ``row_major`` or
    ``column_major``. If this requires relayout of the data, the result is cached in a hidden data
    member for future reuse.
    '''
    data = _get_oa_data(adata, name, layout=layout, formatter=formatter)
    frame = utt.maybe_pandas_frame(data)
    if frame is None:
        frame = utt.to_pandas_frame(utt.to_proper_matrix(data),
                                    index=adata.obs_names,
                                    columns=columns)
    return frame


def get_oa_proper(
    adata: AnnData,
    name: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.ProperMatrix:
    '''
    Same as ``get_oa_data`` but returns a :py:const:`metacells.utilities.typing.ProperMatrix`.
    '''
    data = _get_oa_data(adata, name, layout=layout, formatter=formatter)
    return utt.to_proper_matrix(data, default_layout=layout or 'row_major')


def _get_va_data(
    adata: AnnData,
    name: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.Matrix:
    data = _get_layout_data(adata, 'va', adata.varm,
                            shape=(adata.n_vars, 0),
                            name=name, layout=layout)
    utl.log_get(adata, 'va', name, data, formatter=formatter)
    return data


def get_va_frame(
    adata: AnnData,
    name: Union[str, utt.Matrix],
    *,
    columns: Optional[Collection],
    layout: Optional[str] = None,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.PandasFrame:
    '''
    Get per-variable-per-any (per-cell-per-any) data as a pandas data frame.

    Rows are variables (genes), indexed by their names.
    Columns are "something" - specify ``columns`` to specify an index.

    If ``name`` is a string, it is the name of a per-variable annotation to fetch.
    Otherwise, it should be some matrix of data of the appropriate size.

    If ``layout`` (default: {layout}) is specified, it must be one of ``row_major`` or
    ``column_major``. If this requires relayout of the data, the result is cached in a hidden data
    member for future reuse.
    '''
    data = _get_va_data(adata, name, layout=layout, formatter=formatter)
    frame = utt.maybe_pandas_frame(data)
    if frame is None:
        frame = utt.to_pandas_frame(utt.to_proper_matrix(data),
                                    index=adata.var_names,
                                    columns=columns)
    return frame


def get_va_proper(
    adata: AnnData,
    name: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.ProperMatrix:
    '''
    Same as ``get_va_data`` but returns a :py:const:`metacells.utilities.typing.ProperMatrix`.
    '''
    data = _get_va_data(adata, name, layout=layout, formatter=formatter)
    return utt.to_proper_matrix(data, default_layout=layout or 'row_major')


def _get_vo_data(
    adata: AnnData,
    name: Union[str, utt.Matrix],
    *,
    layout: Optional[str] = None,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.Matrix:
    if name is None:
        name = '__x__'

    data = _get_layout_data(adata, 'vo', adata.layers,
                            shape=(adata.n_obs, adata.n_vars),
                            name=name, layout=layout)
    utl.log_get(adata, 'vo', name, data, formatter=formatter)
    return data


def get_vo_frame(
    adata: AnnData,
    name: Union[str, utt.Matrix] = '__x__',
    *,
    layout: Optional[str] = None,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.PandasFrame:
    '''
    Get per-variable-per-observation (per-gene-per-cell) data as a pandas data frame.

    Rows are observations (cells), indexed by the observation names (typically cell barcode).
    Columns are variables (genes), indexed by their names.

    If ``name`` is a string, it is the name of a per-variable annotation to fetch.
    Otherwise, it should be some matrix of data of the appropriate size.

    If ``layout`` (default: {layout}) is specified, it must be one of ``row_major`` or
    ``column_major``. If this requires relayout of the data, the result is cached in a hidden data
    member for future reuse.
    '''
    data = _get_vo_data(adata, name, layout=layout, formatter=formatter)
    frame = utt.maybe_pandas_frame(data)
    if frame is None:
        frame = utt.to_pandas_frame(utt.to_proper_matrix(data),
                                    index=adata.obs_names,
                                    columns=adata.var_names)
    return frame


def get_vo_proper(
    adata: AnnData,
    name: Union[str, utt.Matrix] = '__x__',
    *,
    layout: Optional[str] = None,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> utt.ProperMatrix:
    '''
    Same as ``get_vo_data`` but returns a :py:const:`metacells.utilities.typing.ProperMatrix`.
    '''
    data = _get_vo_data(adata, name, layout=layout, formatter=formatter)
    return utt.to_proper_matrix(data, default_layout=layout or 'row_major')


def _get_vo_sum_data(
    adata: AnnData,
    per: str,
    name: Union[str, utt.Shaped],
) -> utt.NumpyVector:
    if not isinstance(name, str):
        assert utt.is_2d(name)
        assert name.shape == adata.shape  # type: ignore
        return utc.sum_per(name, per=per)  # type: ignore

    if not hasattr(adata, '__derived__'):
        derived: Dict[str, utt.NumpyVector] = dict()
        setattr(adata, '__derived__', derived)
    else:
        derived = getattr(adata, '__derived__')

    sum_name = f'vo:{name}:sum_per_{per}'
    sum_data = derived.get(sum_name)
    if sum_data is not None:
        return sum_data

    data = get_vo_proper(adata, name, layout=f'{per}_major')
    sum_data = utc.sum_per(data, per=per)
    derived[sum_name] = sum_data
    return sum_data


def _get_layout_data(
    adata: AnnData,
    per: str,
    annotations: Annotations,
    *,
    shape: Tuple[int, ...],
    name: Union[str, utt.Matrix],
    layout: Optional[str],
) -> Any:
    data = _get_shaped_data(adata, per, annotations, shape=shape, name=name)
    if not isinstance(name, str):
        if not utt.is_layout(data, layout):
            assert layout is not None
            data = utc.to_layout(name, layout=layout)
        return data

    if utt.is_layout(data, layout):
        return data

    assert layout in utt.LAYOUT_OF_AXIS
    layout_name = '%s:%s:%s' % (per, name, layout)

    if not hasattr(adata, '__derived__'):
        derived: Dict[str, utt.ProperMatrix] = dict()
        setattr(adata, '__derived__', derived)
    else:
        derived = getattr(adata, '__derived__')

    layout_data = derived.get(layout_name)
    if layout_data is not None:
        assert layout_data.shape == shape
        assert utt.is_layout(layout_data, layout)
    else:
        layout_data = utc.to_layout(data, layout=layout)
        derived[layout_name] = layout_data

    if not utt.frozen(layout_data):
        utt.freeze(layout_data)

    return layout_data


def _get_shaped_data(
    adata: AnnData,
    per: str,
    annotations: Annotations,
    *,
    shape: Tuple[int, ...],
    name: Union[str, utt.Shaped],
) -> Any:
    if isinstance(name, str):
        if per == 'vo' and name == '__x__':
            data = _fix_data(adata.X)
        else:
            if name not in annotations:  # type: ignore
                raise _unknown_data(adata, name, per)
            data = _fix_data(annotations[name])

        if not utt.frozen(data):
            utt.freeze(data)

    else:
        if utt.is_1d(name):
            data = utt.to_numpy_vector(name)
        else:
            data = utt.to_proper_matrix(name)  # type: ignore

    assert data.shape == shape
    return data


def _fix_data(data: Any) -> Any:
    if isinstance(data, ad._core.sparse_dataset.SparseDataset):  # pylint: disable=protected-access
        return data.value
    return data


def _annotation_per(  # pylint: disable=too-many-return-statements
    adata: AnnData, per: str
) -> Annotations:
    if per == 'm':
        return adata.uns
    if per == 'vo':
        return adata.layers
    if per == 'v':
        return adata.var
    if per == 'o':
        return adata.obs
    if per == 'vv':
        return adata.varp
    if per == 'oo':
        return adata.obsp
    if per == 'va':
        return adata.varm
    if per == 'oa':
        return adata.obsm
    raise ValueError('unknown per: %s' % per)


def set_name(adata: AnnData, name: Optional[str]) -> None:
    '''
    Set the ``name`` of the data (for log messages).

    If the name starts with ``.`` it is appended to the current name, if any.
    '''
    if name is None:
        if '__name__' in adata.uns:
            del adata.uns['__name__']
        return

    if name[0] == '.':
        old_name = get_name(adata)
        if old_name is None:
            name = name[1:]
        else:
            name = old_name + name
    adata.uns['__name__'] = name


def get_name(adata: AnnData, default: Optional[str] = None) -> Optional[str]:
    '''
    Return the name of the data (for log messages), if any.

    If no name was set, returns the ``default``.
    '''
    return adata.uns.get('__name__', default)


def set_m_data(
    adata: AnnData,
    name: str,
    data: Any,
    *,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> Any:
    '''
    Set unstructured data.

    If ``formatter`` is specified, its results is used when logging the operation.
    '''
    utl.log_set(adata, 'm', name, data, formatter=formatter)
    adata.uns[name] = data


def set_o_data(
    adata: AnnData,
    name: str,
    data: utt.NumpyVector,
    *,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> Any:
    '''
    Set per-observation (cell) data.

    If ``formatter`` is specified, its results is used when logging the operation.
    '''
    utl.log_set(adata, 'o', name, data, formatter=formatter)

    if not isinstance(data, list):
        utt.mustbe_canonical(data)
        if not utt.frozen(data):
            utt.freeze(data)

    adata.obs[name] = data


def set_v_data(
    adata: AnnData,
    name: str,
    data: utt.NumpyVector,
    *,
    formatter: Optional[Callable[[Any], Any]] = None
) -> Any:
    '''
    Set per-variable (gene) data.

    If ``formatter`` is specified, its results is used when logging the operation.
    '''
    utl.log_set(adata, 'v', name, data, formatter=formatter)

    utt.mustbe_canonical(data)
    if not utt.frozen(data):
        utt.freeze(data)

    adata.var[name] = data


def set_oo_data(
    adata: AnnData,
    name: str,
    data: utt.ProperMatrix,
    *,
    formatter: Optional[Callable[[Any], Any]] = None
) -> Any:
    '''
    Set per-observation-per-observation (cell) data.

    If ``formatter`` is specified, its results is used when logging the operation.
    '''
    utl.log_set(adata, 'oo', name, data, formatter=formatter)

    utt.mustbe_canonical(data)
    if not utt.frozen(data):
        utt.freeze(data)

    adata.obsp[name] = data


def set_vv_data(
    adata: AnnData,
    name: str,
    data: utt.ProperMatrix,
    *,
    formatter: Optional[Callable[[Any], Any]] = None
) -> Any:
    '''
    Set per-variable-per-variable (gene) data.

    If ``formatter`` is specified, its results is used when logging the operation.
    '''
    utl.log_set(adata, 'vv', name, data, formatter=formatter)

    utt.mustbe_canonical(data)
    if not utt.frozen(data):
        utt.freeze(data)

    adata.varp[name] = data


def set_oa_data(
    adata: AnnData,
    name: str,
    data: utt.ProperMatrix,
    *,
    formatter: Optional[Callable[[Any], Any]] = None
) -> Any:
    '''
    Set per-observation-per-any (cell) data.

    If ``formatter`` is specified, its results is used when logging the operation.
    '''
    utl.log_set(adata, 'oa', name, data, formatter=formatter)

    utt.mustbe_canonical(data)
    if not utt.frozen(data):
        utt.freeze(data)

    adata.obsm[name] = data


def set_va_data(
    adata: AnnData,
    name: str,
    data: utt.ProperMatrix,
    *,
    formatter: Optional[Callable[[Any], Any]] = None
) -> Any:
    '''
    Set per-variable-per-any (gene) data.

    If ``formatter`` is specified, its results is used when logging the operation.
    '''
    utl.log_set(adata, 'va', name, data, formatter=formatter)

    utt.mustbe_canonical(data)
    if not utt.frozen(data):
        utt.freeze(data)

    adata.varm[name] = data


def set_vo_data(
    adata: AnnData,
    name: str,
    data: utt.ProperMatrix,
    *,
    formatter: Optional[Callable[[Any], Any]] = None,
) -> Any:
    '''
    Set per-variable-per-observation (per-gene-per-cell) data.
    '''
    utl.log_set(adata, 'vo', name, data, formatter=formatter)

    utt.mustbe_canonical(data)
    if not utt.frozen(data):
        utt.freeze(data)

    if name == '__x__':
        adata.X = data
    else:
        adata.layers[name] = data


def _unknown_data(adata: AnnData, name: str, per: Optional[str] = None) -> KeyError:
    texts = ['unknown']

    if per is not None:
        texts.append(' ')
        texts.append(per)

    texts.append(' data')

    data_name = get_name(adata)
    if data_name is not None:
        texts.append(': ')
        texts.append(data_name)

    texts.append(' name: ')
    texts.append(name)
    return KeyError(''.join(texts))
