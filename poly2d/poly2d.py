#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
from __future__ import division
import warnings
from math import sqrt
import numpy as np
from numpy.linalg import lstsq
if np.__version__ < '1.4.0':
    from numpy.lib.getlimits import finfo
else:
    from numpy.core.getlimits import finfo

import _poly2d


__all__ = ['poly2d', 'polyfit2d']


polyfunc = {
    1: _poly2d.poly1,
    2: _poly2d.poly2,
    3: _poly2d.poly3,
    4: _poly2d.poly4,
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
    n : int
        The number of the coefficients.
    """
    return 3 + (order-1)*(order+4) // 2


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
    """
    return int((sqrt(8*ncoeffs+1) - 3) // 2)


def _kernel_mat(x, y, m):
    """Return the kernel matrix `X`.

    [1, x[0], y[0], ... , x[0]*y[0]**(n-1), y[0]**n]
    [1, x[1], y[1], ... , x[1]*y[1]**(M-1), y[1]**n]
    ...
    ...
    [1, x[M], y[M], ... , x[M]*y[M]**(n-1), y[M]**n]

    Parameters
    ----------
    x : ndarray
        The x-coordinate of M sample points.
    y : ndarray
        The y-coordinate of M sample points.
    m : int
        The order of the polynomial.

    Returns
    -------
    X : ndarray
        The kernel matrix.

    Raises
    ------
    ValueError
        If m is larger than 4.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    X = np.ones((x.size, ncoeffs(m)), x.dtype)

    if m == 1:
        X[:,1] = x
        X[:,2] = y
    elif m == 2:
        X[:,1] = x
        X[:,2] = y
        X[:,3] = x*x
        X[:,4] = x*y
        X[:,5] = y*y
    elif m == 3:
        X[:,1] = x
        X[:,2] = y
        X[:,3] = x*x
        X[:,4] = x*y
        X[:,5] = y*y
        X[:,6] = X[:,3]*x        # x^3
        X[:,7] = X[:,3]*y        # x^2 y
        X[:,8] = x*X[:,5]        # x y^2
        X[:,9] = y*X[:,5]        # y^3
        X[:,6] = X[:,3]*x
    elif m == 4:
        X[:,1]  = x
        X[:,2]  = y
        X[:,3]  = x*x
        X[:,4]  = x*y
        X[:,5]  = y*y
        X[:,6]  = X[:,3]*x       # x^3
        X[:,7]  = X[:,3]*y       # x^2 y
        X[:,8]  = x*X[:,5]       # x y^2
        X[:,9]  = y*X[:,5]       # y^3
        X[:,10] = X[:,3]*X[:,3]  # x^4
        X[:,11] = X[:,6]*y       # x^3 y
        X[:,12] = X[:,3]*X[:,5]  # x^2 y^2
        X[:,13] = x*X[:,9]       # x y^3
        X[:,14] = X[:,5]*X[:,5]  # y^4
    else:
        raise ValueError('Invalid polynomial order: {0}'.format(m))
    return X


def polyfit2d(x, y, xt, yt, order=1, rcond=None, full_output=False):
    """Return the polynomial coefficients determined by the least square fit.

    x' = cx[0] + cx[1]*x + cx[2]*y + ... + cx[n-1]*x*y**(n-1) + cx[n]*y**n
    y' = cy[0] + cy[1]*x + cy[2]*y + ... + cy[n-1]*x*y**(n-1) + cy[n]*y**n

    Parameters
    ----------
    x : array_like, shape (M,)
        The x-coordinates of the M sample points.
    y : array_like, shape (M,)
        The y-coordinates of the M sample points.
    xt : array_like, shape (M,)
        The x'-coordinates of the M sample points.
    yt : array_like, shape (M,)
        The y'-coordinates of the M sample points.

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
    cx, cy : ndarray
        The polynomial coefficients in increasing powers.

    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient.  The warning is only raised if `full_output` = False.

    See Also
    --------
    numpy.linalg.lstsq
    numpy.lib.polynomial.poolyfit
    """
    x = np.asarray(x, np.float)
    y = np.asarray(y, np.float)
    xt = np.asarray(xt, np.float)
    yt = np.asarray(yt, np.float)

    # -- Check inputs.
    if x.size != y.size or x.size != xt.size or x.size != yt.size:
        raise ValueError("`x`, `y`, `xt`, and `yt` must have the same size.")
    if x.ndim != 1:
        raise ValueError("`x` must be 1-dim.")
    if y.ndim != 1:
        raise ValueError("`y` must be 1-dim.")
    if xt.ndim != 1:
        raise ValueError("`xt` must be 1-dim.")
    if yt.ndim != 1:
        raise ValueError("`yt` must be 1-dim.")

    # -- Set `rcond`
    if rcond is None:
        rcond = x.size * finfo(x.dtype).eps

    # -- Scale `x` and `y`.
    scale = max(abs(x.max()), abs(y.max()))
    if scale != 0:
        x /= scale
        y /= scale

    # -- Solve the least square equations.
    v = _kernel_mat(x, y, order)
    cx, resids_x, rank, s_x = lstsq(v, xt)
    cy, resids_y, rank, s_y = lstsq(v, yt)

    # -- Warn on rank deficit, which indicates an ill-conditioned matrix.
    if rank != ncoeffs(order) and not full_output:
        msg = "`polyfit2d` may be poorly conditioned"
        warnings.warn(msg, RankWarning)

    # -- Scale the returned coefficients.
    if scale != 0:
        S = _kernel_mat([scale], [scale], order)[0]
        cx /= S
        cy /= S

    if full_output:
        return cx, cy, resids_x, resids_y, s_x, s_y, rank, rcond
    else:
        return cx, cy


class poly2d(object):
    """A 2-dimensional polynomial transform class.

    x' = cx[0] + cx[1]*x + cx[2]*y + ... + cx[n-1]*x*y**(n-1) + cx[n]*y**n
    y' = cy[0] + cy[1]*x + cy[2]*y + ... + cy[n-1]*x*y**(n-1) + cy[n]*y**n

    Parameters
    ----------
    cx : array_like
        The polynomial coefficients for the x'-coordinate 'in
        *increasing* powers.
    cy : array_like
        The polynomial coefficients for the y'-coordinate 'in
        *increasing* powers.

    Attributes
    ----------
    cx : array_like
        The polynomial coefficients for the x'-coordinate 'in
        *increasing* powers.
    cy : array_like
        The polynomial coefficients for the y'-coordinate 'in
        *increasing* powers.
    order : int
        The order of the polynomial.
    valid_ncoeffs : tuple of ints
        The valid number of the coefficients per coordinate.
    """
    max_order = len(polyfunc)
    valid_ncoeffs = tuple(ncoeffs(m) for m in range(1, max_order+1))

    def __init__(self, cx, cy):
        """
        Parameters
        ----------
        cx : array_like
            The polynomial coefficients for the x'-coordinate 'in
            *increasing* powers.
        cy : array_like
            The polynomial coefficients for the y'-coordinate 'in
            *increasing* powers.

        Raises
        ------
        ValueError
            If `cx` or `cy` are not consistent.
        """
        cx = np.atleast_1d(cx) + 0.0
        cy = np.atleast_1d(cy) + 0.0

        # -- Check the inputs.
        if cx.size != cy.size:
            raise ValueError("`cx` and `cy` must have the same size.")
        if cx.ndim != 1:
            raise ValueError("`cx` must be 1-dim.")
        if cy.ndim != 1:
            raise ValueError("`cy` must be 1-dim.")
        if len(cx) not in self.valid_ncoeffs:
            msg = ('Invalid number of the coefficients: {0}\n'
                   'Must be one of {1}'.format(len(cx), str(self.valid_ncoeffs)))
            raise ValueError(msg)

        self._cx = cx
        self._cy = cy

        self._order = order(len(self._cx))

    def __call__(self, x, y):
        """Return the transformed coordinates.

        Parameters
        ----------
        x : array_like, shape(M,)
            The x-coordinate.
        y : array_like, shape(M,)
            The y-coordinate.

        Returns
        -------
        xt : ndarray, shape(M,)
            The transformed x-coordinate.
        yt : ndarray, shape(M,)
            The transformed y-coordinate.

        Raises
        ------
        ValueError
            If `x` or `y` are not consistent.
        """
        x = np.atleast_1d(x) + 0.0
        y = np.atleast_1d(y) + 0.0

        # -- Chek inputs.
        if x.size != y.size:
            raise ValueError('`x` and `y` must have the same size.')
        if x.ndim != 1:
            raise ValueError("`x` must be 1-dim.")
        if y.ndim != 1:
            raise ValueError("`y` must be 1-dim.")

        return polyfunc[self._order](x, y, self._cx, self._cy)

    @property
    def order(self):
        """The order of the polynomial."""
        return self._order

    @property
    def cx(self):
        """The polynomial coefficients for the x'-coordinate."""
        return self._cx

    @property
    def cy(self):
        """The polynomial coefficients for the y'-coordinate."""
        return self._cy
