#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
from __future__ import division
import warnings
from math import sqrt
import numpy as np
from numpy.linalg import lstsq

import _poly2d


__all__ = ['poly2d', 'polyfit2d', 'poly2d_transform', 'polyfit2d_transform']


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


def num_coeffs(order):
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
    """
    return (order+1)*(order+2) // 2


def order(num_coeffs):
    """Return the polynomial order for a given number of the
    coefficients.

    Parameters
    ----------
    num_coeffs : int
        The number of the coeefficients.

    Returns
    -------
    order : int
        The polynomial order.
    """
    return int((sqrt(8*num_coeffs+1) - 3) // 2)


def vandermonde(x, y, order):
    """Return the Vandermonde matrix for a given order.

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
    V : ndarray
        The Vandermonde matrix.

    Raises
    ------
    ValueError
        If m is larger than 4.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = (order+1)*(order+2) // 2
    V = np.zeros((x.size, n), x.dtype)

    V[:,0] = np.ones(x.size)
    i = 1
    for o in range(1,order+1):
        V[:,i:i+o] = x[:,np.newaxis] * V[:,i-o:i]
        V[:,i+o] = y * V[:,i-1]
        i += o + 1
    return V


def polyfit2d(x, y, z, order=1, rcond=None, full_output=False):
    """Return the polynomial coefficients determined by the least square fit.

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
        The polynomial coefficients in ascending powers.

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

    ## Check inputs.
    if x.size != y.size or x.size != z.size:
        raise ValueError("`x`, `y`, and `z` must have the same size.")
    if x.ndim != 1:
        raise ValueError("`x` must be 1-dim.")
    if y.ndim != 1:
        raise ValueError("`y` must be 1-dim.")
    if z.ndim != 1:
        raise ValueError("`z` must be 1-dim.")

    ## Set `rcond`
    if rcond is None:
        rcond = x.size * np.finfo(x.dtype).eps

    ## Scale `x` and `y`.
    scale = max(abs(x.max()), abs(y.max()))
    if scale != 0:
        x /= scale
        y /= scale

    ## Solve the least square equations.
    v = vandermonde(x, y, order)
    c, resids, rank, s = lstsq(v, z)

    ## Warn on rank deficit, which indicates an ill-conditioned matrix.
    if rank != num_coeffs(order) and not full_output:
        msg = "`polyfit2d` may be poorly conditioned"
        warnings.warn(msg, RankWarning)

    ## Scale the returned coefficients.
    if scale != 0:
        S = vandermonde([scale], [scale], order)[0]
        c /= S

    if full_output:
        return c, c, resids, s, rank, rcond
    else:
        return c


def polyfit2d_transform(x, y, xt, yt, order=1, rcond=None, full_output=False):
    """Return the polynomial coefficients determined by the least square fit.

    x' = cx[0] + cx[1]*x + cx[2]*y + ... + cx[n-1]*x*y**(n-1) + cx[n]*y**n
    y' = cx[0] + cx[1]*x + cx[2]*y + ... + cx[n-1]*x*y**(n-1) + cx[n]*y**n

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
        The polynomial coefficients in ascending powers.

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
    xt = np.asarray(xt, np.float64)
    yt = np.asarray(yt, np.float64)

    ## Check inputs.
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

    ## Set `rcond`
    if rcond is None:
        rcond = x.size * np.finfo(x.dtype).eps

    ## Scale `x` and `y`.
    scale = max(abs(x.max()), abs(y.max()))
    if scale != 0:
        x /= scale
        y /= scale

    ## Solve the least square equations.
    v = vandermonde(x, y, order)
    cx, resids_x, rank, s_x = lstsq(v, xt)
    cy, resids_y, rank, s_y = lstsq(v, yt)

    ## Warn on rank deficit, which indicates an ill-conditioned matrix.
    if rank != num_coeffs(order) and not full_output:
        msg = "`polyfit2d` may be poorly conditioned"
        warnings.warn(msg, RankWarning)

    ## Scale the returned coefficients.
    if scale != 0:
        S = vandermonde([scale], [scale], order)[0]
        cx /= S
        cy /= S

    if full_output:
        return cx, cy, resids_x, resids_y, s_x, s_y, rank, rcond
    else:
        return cx, cy


class _poly2d(object):
    @property
    def order(self):
        """The order of the polynomial."""
        return self._order


class poly2d(_poly2d):
    """A 2-dimensional polynomial class.

    z = c[0] + c[1]*x + c[2]*y + ... + c[n-1]*x*y**(n-1) + c[n]*y**n

    Parameters
    ----------
    c : array_like
        The polynomial coefficients in *ascending* powers.

    Attributes
    ----------
    c : array_like
        The polynomial coefficients in *ascending* powers.
    order : int
        The order of the polynomial.
    """
    def __init__(self, c):
        """
        Parameters
        ----------
        c : array_like
            The polynomial coefficients for the x'-coordinate 'in
            *ascending* powers.

        Raises
        ------
        ValueError
        """
        c = np.atleast_1d(c) + 0.0
        ## Check the inputs.
        if c.ndim != 1:
            raise ValueError("`c` must be 1-dim.")

        self._c = c
        self._order = order(len(self._c))

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

        ## Chek inputs.
        if x.size != y.size:
            raise ValueError('`x` and `y` must have the same size.')
        if x.ndim != 1:
            raise ValueError("`x` must be 1-dim.")
        if y.ndim != 1:
            raise ValueError("`y` must be 1-dim.")

        try:
            return _polyfunc[self._order](x, y, self._c)
        except KeyError:
            V = vandermonde(x, y, self._order)
            return V * self._c

    @property
    def c(self):
        """The polynomial coefficients for the x'-coordinate."""
        return self._c


class poly2d_transform(_poly2d):
    """A 2-dimensional polynomial transform class.

    x' = cx[0] + cx[1]*x + cx[2]*y + ... + cx[n-1]*x*y**(n-1) + cx[n]*y**n
    y' = cy[0] + cy[1]*x + cy[2]*y + ... + cy[n-1]*x*y**(n-1) + cy[n]*y**n

    Parameters
    ----------
    cx : array_like
        The polynomial coefficients for the x'-coordinate 'in
        *ascending* powers.
    cy : array_like
        The polynomial coefficients for the y'-coordinate 'in
        *ascending* powers.

    Attributes
    ----------
    cx : array_like
        The polynomial coefficients for the x'-coordinate 'in
        *ascending* powers.
    cy : array_like
        The polynomial coefficients for the y'-coordinate 'in
        *ascending* powers.
    order : int
        The order of the polynomial.
    """
    def __init__(self, cx, cy):
        """
        Parameters
        ----------
        cx : array_like
            The polynomial coefficients for the x'-coordinate 'in
            *ascending* powers.
        cy : array_like
            The polynomial coefficients for the y'-coordinate 'in
            *ascending* powers.

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

        try:
            xt = _polyfunc[self._order](x, y, self._cx)
            yt = _polyfunc[self._order](x, y, self._cy)
        except KeyError:
            V = vandermonde(x, y, self._order)
            xt =  V * self._cx
            yt =  V * self._cy
        return xt, yt

    @property
    def cx(self):
        """The polynomial coefficients for the x'-coordinate."""
        return self._cx

    @property
    def cy(self):
        """The polynomial coefficients for the y'-coordinate."""
        return self._cy
