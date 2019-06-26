# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: skip-file

"""Doc placeholder for numpy ops with prefix _np."""


def _np_reshape(a, newshape, order='C'):
    """Gives a new shape to an array without changing its data.

    Parameters
    ----------
    a : ndarray
        Array to be reshaped.
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.
    order : {'C'}, optional
        Read the elements of `a` using this index order, and place the
        elements into the reshaped array using this index order.  'C'
        means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest. Other order types such as 'F'/'A'
        may be added in the future.

    Returns
    -------
    reshaped_array : ndarray
        It will be always a copy of the original array. This behavior is different
        from the official NumPy package where views of the original array may be
        generated.

    See Also
    --------
    ndarray.reshape : Equivalent method.
    """
    pass


def _np_ones_like(a):
    """Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : ndarray
        The shape and data-type of `a` define these same attributes of
        the returned array.

    Returns
    -------
    out : ndarray
        Array of ones with the same shape and type as `a`.
    """
    pass


def _np_zeros_like(a, dtype=None, **kwargs):
    """
    zeros_like(a, dtype=None, order='C', subok=True)

    Return an array of zeros with the same shape as a given array.

    Parameters
    ----------
    a : ndarray
        The shape of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
        .. versionadded:: 1.6.0
    order : {'C'}, optional, default: C
        Store multi-dimensional data in row-major
        (C-style) in memory. Note that the column-major is not supported yet.
    ctx : Context, optional
        An optional device context (default is the current default context).
    
    Returns
    -------
    out : ndarray
        Array of zeros with the same shape as `a`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    zeros : Return a new array setting values to zero.

    Examples
    --------
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]])
    >>> y = np.arange(3, dtype=float)
    >>> y
    array([0., 1., 2.])
    >>> np.zeros_like(y)
    array([0.,  0.,  0.])

    Notes
    -----

    This function differs to the original `numpy.ones
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros_like.html>`_ in
    the following aspects:
    - The parameter `dtype` and `subok` are not supported now.
    - Only row-major is supported.
    - There is an additional `ctx` argument to specify the device, e.g. the i-th
      GPU.
    """
    pass


def _np_repeat(a, repeats, axis=None):
    """Repeat elements of an array.

    Parameters
    ----------
    a : ndarray
        Input array.
    repeats : int or array of ints
        The number of repetitions for each element.  `repeats` is broadcasted
        to fit the shape of the given axis.
    axis : int, optional
        The axis along which to repeat values.  By default, use the
        flattened input array, and return a flat output array.

    Returns
    -------
    repeated_array : ndarray
        Output array which has the same shape as `a`, except along
        the given axis.
    """
    pass


def _npi_multinomial(a):
    """Draw samples from a multinomial distribution.

    The multinomial distribution is a multivariate generalisation of the binomial distribution.
    Take an experiment with one of ``p`` possible outcomes. An example of such an experiment is throwing a dice,
    where the outcome can be 1 through 6. Each sample drawn from the distribution represents n such experiments.
    Its values, ``X_i = [X_0, X_1, ..., X_p]``, represent the number of times the outcome was ``i``.


    Parameters
    ----------
    n : int
        Number of experiments.
    pvals : sequence of floats, length p
        Probabilities of each of the p different outcomes. These should sum to 1
        (however, the last element is always assumed to account for the remaining
        probability, as long as ``sum(pvals[:-1]) <= 1)``.
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k`` sam-
        ples are drawn. Default is None, in which case a single value is returned.

    Returns
    -------
    out : ndarray
        The drawn samples, of shape size, if that was provided. If not, the shape is ``(N,)``.
        In other words, each entry ``out[i,j,...,:]`` is an N-dimensional value drawn from the distribution.
    """
    pass


def _npi_linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, **kwargs):  # pylint: disable=too-many-arguments
    """Return evenly spaced numbers over a specified interval.

    Returns num evenly spaced samples, calculated over the interval [start, stop].
    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : array_like
        The starting value of the sequence.
    stop : array_like
        The end value of the sequence, unless endpoint is set to False. In
        that case, the sequence consists of all but the last of num + 1
        evenly spaced samples, so that stop is excluded. Note that the step
        size changes when endpoint is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, stop is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (samples, step), where step is the spacing between samples.
    dtype : dtype, optional
        The type of the output array. If dtype is not given, infer the data
        type from the other input arguments.
    axis : int, optional
        The axis in the result to store the samples. Relevant only if start or
        stop are array-like. By default (0), the samples will be along a new
        axis inserted at the beginning. Use -1 to get an axis at the end.

    Returns
    -------
    samples : ndarray
        There are num equally spaced samples in the closed interval
        `[start, stop]` or the half-open interval `[start, stop)`
        (depending on whether endpoint is True or False).
    step : float, optional
        Only returned if retstep is True
        Size of spacing between samples.

    Notes
    -----
    'start' and 'stop' do not support list, numpy ndarray and mxnet ndarray
    axis could only be 0
    """
    pass


def _np_cumsum(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis along which the cumulative sum is computed. The default
        (None) is to compute the cumsum over the flattened array.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed.  If `dtype` is not specified, it defaults
        to the dtype of `a`, unless `a` has an integer dtype with a
        precision less than that of the default platform integer.  In
        that case, the default platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary. See `doc.ufuncs`
        (Section "Output arguments") for more details.

    Returns
    -------
    cumsum_along_axis : ndarray.
        A new array holding the result is returned unless `out` is
        specified, in which case a reference to `out` is returned. The
        result has the same size as `a`, and the same shape as `a` if
        `axis` is not None or `a` is a 1-d array.
    """
    pass


def reciprocal(x, out=None, **kwargs):
    """
    Return the reciprocal of the argument, element-wise.
    Calculates ``1/x``.

    Parameters
    ----------
    x : array_like
        Input array.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have 
        a shape that the inputs broadcast to. If not provided or None, 
        a freshly-allocated array is returned. A tuple 
        (possible only as a keyword argument) must have length equal to 
        the number of outputs.
    
    Returns
    -------
    y : ndarray
        Return array. This is a scalar if x is a scalar.
    
    Notes
    -----
    .. note::
        This function is not designed to work with integers.
    For integer arguments with absolute value larger than 1 the result is
    always zero because of the way Python handles integer division.  For
    integer zero the result is an overflow.

    Notes
    -----
    Only support ndarray now.
    """
    return _mx_nd_np.reciprocal(x, out=out, **kwargs)
