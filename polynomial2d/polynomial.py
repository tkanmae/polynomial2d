#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import warnings
import numpy as np

from . import _poly2d


__all__ = ['polyval2d', 'polyfit2d']


# Use the polynomial functions in the C-extension module `_poly2d` if
# a polynomial order is not greater than 5.  These functions are mush
# faster than calculation using a Vandermonde matrix.
_polyfunc = {
    1: _poly2d.poly1,
    2: _poly2d.poly2,
    3: _poly2d.poly3,
    4: _poly2d.poly4,
    5: _poly2d.poly5,
}


class RankWarning(UserWarning):
    pass


def _polyvander2d(x, y, deg):
    """Return the pseudo-Vandermonde matrix for a given degree.

    [1, x[0], y[0], ... , x[0]*y[0]**(deg-1), y[0]**deg]
    [1, x[1], y[1], ... , x[1]*y[1]**(deg-1), y[1]**deg]
    ...                                          ...
    ...                                          ...
    [1, x[M], y[M], ... , x[M]*y[M]**(deg-1), y[M]**deg]

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinate of M sample points.
    y : array_like, shape (M,)
        x-coordinate of M sample points.
    deg : int
        Degree of the polynomial.

    Returns
    -------
    V : ndarray
        The Vandermonde matrix.
    """
    x = np.array(x, copy=False, ndmin=1) + 0.0
    y = np.array(y, copy=False, ndmin=1) + 0.0
    if x.ndim != 1:
        raise ValueError("x must be 1-dim.")
    if y.ndim != 1:
        raise ValueError("y must be 1-dim.")

    dims = ((deg+1)*(deg+2) // 2, ) + x.shape
    v = np.empty(dims, dtype=x.dtype)
    v[0] = x * 0 + 1.0
    i = 1
    for j in range(1, deg+1):
        v[i:i+j] = x * v[i-j:i]
        v[i+j] = y * v[i-1]
        i += j + 1
    return np.rollaxis(v, 0, v.ndim)


def polyfit2d(x, y, z, deg=1, rcond=None, full_output=False):
    """Return the polynomial coefficients determined by the least-square fit.

    z = c_00 + c_10 * x + c_01 * y + ... + c_0n * y^n

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points.
    y : array_like, shape (M,)
        y-coordinates of the M sample points.
    z : array_like, shape (M,)
        z-coordinates of the M sample points.
    deg : int, optional
        Degree of the polynomial to be fit.
    rcond : float, optional
        Relative condition of the fit.  Singular values smaller than
        `rcond`, relative to the largest singular value, will be
        ignored.  The default value is ``len(x)*eps``, where `eps` is
        the relative precision of the platform's float type, about 2e-16
        in most cases.
    full_output : {True, False}, optional
        Just the coefficients are returned if False, and diagnostic
        information from the SVD is also returned if True.

    Returns
    -------
    coef : ndarray
        Polynomial coefficients.
    [residuals, rank, singular_values, rcond] : if `full_output` = True
        Sum of the squared residuals of the least-squares fit; the
        effective rank of the scaled pseudo-Vandermonde matrix; its
        singular values, and the specified value of `rcond`. For more
        details, see `numpy.linalg.lstsq`.

    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient.  The warning is only raised if `full_output` = False.

    See Also
    --------
    numpy.linalg.lstsq
    numpy.lib.polynomial.polyfit
    """
    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0
    z = np.asarray(z) + 0.0

    deg = int(deg)
    if deg < 1:
        raise ValueError("deg must be larger than 1.")

    # Check inputs.
    if x.ndim != 1:
        raise ValueError("x must be 1-dim.")
    if y.ndim != 1:
        raise ValueError("y must be 1-dim.")
    if z.ndim != 1:
        raise ValueError("z must be 1-dim.")
    if x.size != y.size or x.size != z.size:
        raise ValueError("x, y, and z must have the same size.")

    lhs = _polyvander2d(x, y, deg).T
    rhs = z.T

    # Set rcond.
    if rcond is None:
        rcond = x.size * np.finfo(x.dtype).eps

    # Determine the norms of the design maxtirx columns.
    if issubclass(lhs.dtype.type, np.complexfloating):
        scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
    else:
        scl = np.sqrt(np.square(lhs).sum(1))
    scl[scl == 0] = 1

    # Solve the least squares problem.
    c, resids, rank, s = np.linalg.lstsq(lhs.T / scl, rhs.T, rcond)
    c = (c.T / scl).T

    # Warn on rank reduction.
    if rank != lhs.shape[0] and not full_output:
        msg = "The fit may be poorly conditioned."
        warnings.warn(msg, RankWarning)

    inds = []
    for m in range(deg + 1):
        for j in range(m + 1):
            for i in range(m + 1):
                if i + j != m:
                    continue
                inds.append((i, j))
    cnew = np.zeros((deg + 1, deg + 1))
    cnew[zip(*inds)] = c

    if full_output:
        return cnew, [resids, rank, s, rcond]
    else:
        return cnew


def polyval2d(x, y, c):
    """Evaluate a 2-D polynomial at points (x, y).

    This function returns the value

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * x^i * y^j

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points (x, y), where x
        and y must have the same shape. If x or y is a list or tuple, it is
        first converted to an ndarray, otherwise it is left unchanged and, if it
        is not an ndarray, it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j is contained in `c[i,j]`.

    Returnes
    --------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points formed with pairs
        of corresponding values from x and y.

    See Also
    --------
    numpy.polynomial.polynomial.polyval2d
    """
    from numpy.polynomial.polynomial import polyval2d

    c = np.asarray(c)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError('c must be a squared 2-dim array.')
    return polyval2d(x, y, c)
