#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import warnings
from math import sqrt
import numpy as np
from numpy.linalg import lstsq

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
    """Issued by `polyfit2d` when the matrix `X` is rank deficient."""
    pass


def ncoeffs(order):
    """Return the number of the coefficients for a given polynomial
    order.

    Parameters
    ----------
    order : int
        The polynomial order.

    Returns
    -------
    num : int
        The number of the coefficients.

    Raises
    ------
    ValueError
        If `order` is less than 1.
    """
    if not isinstance(order, int):
        raise TypeError("`order` must be a int")
    if order < 1:
        raise ValueError("`order` must be greater than 0: {0}".format(order))
    return (order+1)*(order+2) // 2


def order(ncoeffs):
    """Return the polynomial order for a given number of the
    coefficients.

    Parameters
    ----------
    ncoeffs : int
        The number of the coeefficients.

    Returns
    -------
    order : int
        The polynomial order.

    Raises
    ------
    ValueError
        If `ncoeffs` is not consistent with any of 2-dim polynomial orders.
    """
    order = (sqrt(8*ncoeffs+1) - 3) / 2
    if order % 1 > 1e-8 or order < 1:
        raise ValueError("`ncoeff` is not consistent with any of 2-dim"
                         "polynomial orders: {0}".format(ncoeffs))
    return int(order)


def vandermonde(x, y, order):
    """Return the Vandermonde matrix for a given order.

    [1, x[0], y[0], ... , x[0]*y[0]**(n-1), y[0]**n]
    [1, x[1], y[1], ... , x[1]*y[1]**(n-1), y[1]**n]
    ...                                          ...
    ...                                          ...
    [1, x[M], y[M], ... , x[M]*y[M]**(n-1), y[M]**n]

    Parameters
    ----------
    x : array_like, shape (M,)
        The x-coordinate of M sample points.
    y : array_like, shape (M,)
        The y-coordinate of M sample points.
    order : int
        The order of the polynomial.

    Returns
    -------
    V : ndarray
        The Vandermonde matrix.
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    if x.ndim != 1:
        raise ValueError("`x` must be 1-dim.")
    if y.ndim != 1:
        raise ValueError("`y` must be 1-dim.")

    n = (order+1)*(order+2) // 2
    V = np.zeros((x.size, n), x.dtype)

    V[:, 0] = np.ones(x.size)
    i = 1
    for o in range(1, order+1):
        V[:, i:i+o] = x[:, np.newaxis] * V[:, i-o:i]
        V[:, i+o] = y * V[:, i-1]
        i += o + 1
    return V


def polyfit2d(x, y, z, order=1, rcond=None, full_output=False):
    """Return the polynomial coefficients determined by the least-square fit.

    z = cx[0] + cx[1]*x + cx[2]*y + ... + cx[n-1]*x*y**(n-1) + cx[n]*y**n

    Parameters
    ----------
    x : array_like, shape (M,)
        The x-coordinates of the M sample points.
    y : array_like, shape (M,)
        The y-coordinates of the M sample points.
    z : array_like, shape (M,)
        The z-coordinates of the M sample points.

    order : int, optional
        The order of the fitting polynomial.
    rcond : float, optional
        The relative condition of the fit.  Singular values smaller than
        this number relative to the largest singular value will be
        ignored.
    full_output : {True, False}, optional
        Just the coefficients are returned if False, and diagnostic
        information from the SVD is also returned if True.

    Returns
    -------
    c : ndarray
        The polynomial coefficients in *ascending* powers.
    residuals, rank, singular_values, rcond : if `full_output` = True
        Residuals of the least-squares fit, the effective rank of the scaled
        Vandermonde coefficient matrix, its singular values, and the specified
        value of `rcond`. For more details, see `numpy.linalg.lstsq`.

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
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    z = np.asarray(z, np.float64)

    # Check inputs.
    if x.size != y.size or x.size != z.size:
        raise ValueError("`x`, `y`, and `z` must have the same size.")
    if x.ndim != 1:
        raise ValueError("`x` must be 1-dim.")
    if y.ndim != 1:
        raise ValueError("`y` must be 1-dim.")
    if z.ndim != 1:
        raise ValueError("`z` must be 1-dim.")
    # Set `rcond`
    if rcond is None:
        rcond = x.size * np.finfo(x.dtype).eps
    # Scale `x` and `y`.
    scale = max(abs(x.max()), abs(y.max()))
    if scale != 0:
        x /= scale
        y /= scale

    # Solve the least square equations.
    v = vandermonde(x, y, order)
    c, rsq, rank, s = lstsq(v, z)

    # Warn on rank deficit, which indicates an ill-conditioned matrix.
    if rank != ncoeffs(order) and not full_output:
        msg = "`polyfit2d` may be poorly conditioned"
        warnings.warn(msg, RankWarning)
    # Scale the returned coefficients.
    if scale != 0:
        S = vandermonde([scale], [scale], order)[0]
        c /= S

    if full_output:
        return c, rsq, s, rank, rcond
    else:
        return c


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
