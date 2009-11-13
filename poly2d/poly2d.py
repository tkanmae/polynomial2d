# -*- coding: utf-8 -*-
from __future__ import division
import warnings
import numpy as np

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

def order_to_ncoeffs(order):
    """Return the number of the coefficients per coordinate for a given
    polynomial order.

    Parameters
    ----------
    order : int
        The polynomial order.

    Returns
    -------
    ncoeffs : int
        The number of the coeefficients per coordinate.
    """
    return 3 + (order-1)*(order+4) // 2

def ncoeffs_to_order(order):
    """Return the polynomial order for a given number of the coefficients
    per coordinate.

    Parameters
    ----------
    ncoeffs : int
        The number of the coeefficients per coordinate.

    Returns
    -------
    order : int
        The polynomial order.
    """
    from math import sqrt
    return int((sqrt(8*order+1) - 3) / 2)

def fitting_matrix(x, y, order):
    """Return the fitting matrix `X`.

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
    order : int
        The order of the polynomial.

    Returns
    -------
    X : ndarray
        The fitting matrix.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    X = np.ones((x.size, order_to_ncoeffs(order)), x.dtype)
    if order == 1:
        X[:,1] = x
        X[:,2] = y
    elif order == 2:
        X[:,1] = x
        X[:,2] = y
        X[:,3] = x*x
        X[:,4] = x*y
        X[:,5] = y*y
    elif order == 3:
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
    elif order == 4:
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
        raise ValueError('Invalid polynomial order: %d' % order)
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
        deficient. The warning is only raised if `full_output` = False.

    See Also
    --------
    numpy.linalg.lstsq
    numpy.lib.polynomial.poolyfit
    """
    from numpy.lib.getlimits import finfo
    from numpy.linalg import lstsq

    x  = np.asarray(x, float)
    y  = np.asarray(y, float)
    xt = np.asarray(xt, float)
    yt = np.asarray(yt, float)

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
    v = fitting_matrix(x, y, order)
    cx, resids_x, rank, s_x = lstsq(v, xt)
    cy, resids_y, rank, s_y = lstsq(v, yt)

    # -- Warn on rank deficit, which indicates an ill-conditioned matrix.
    if rank != order_to_ncoeffs(order) and not full_output:
        msg = "`polyfit2d` may be poorly conditioned"
        warnings.warn(msg, RankWarning)

    # -- Scale the returned coefficients.
    if scale != 0:
        S = fitting_matrix([scale], [scale], order)[0]
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
    cx, cy: array_like
        The polynomial coefficients in increasing powers.

    Attributes
    ----------
    order
    cx : ndarray
        The polynomial coefficients for the x'-coordinate.
    cy : ndarray
        The polynomial coefficients for the y'-coordinate.
    order : int
        The order of the polynomial.
    valid_ncoeffs : tuple
        The valid number of the coefficients per coordinate.
    """
    maximum_order = len(polyfunc)
    valid_ncoeffs = tuple(order_to_ncoeffs(n) for n in range(1, maximum_order+1))

    def __init__(self, cx, cy):
        cx = np.atleast_1d(cx) + 0.0
        cy = np.atleast_1d(cy) + 0.0

        # -- Check inputs.
        if cx.size != cy.size:
            raise ValueError("`cx` and `cy` must have the same size.")
        if cx.ndim != 1:
            raise ValueError("`cx` must be 1-dim.")
        if cy.ndim != 1:
            raise ValueError("`cy` must be 1-dim.")
        if len(cx) not in self.valid_ncoeffs:
            msg = 'Invalid number of the coefficients: %d\n' % len(cx) + \
                  'Must be one of ' + str(self.valid_ncoeffs)
            raise ValueError(msg)

        self.cx = cx
        self.cy = cy

        self._order = ncoeffs_to_order(len(cx))

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

        return polyfunc[self.order](x, y, self.cx, self.cy)

    @property
    def order(self):
        """The order of the polynomial."""
        return self._order
