# -*- coding: utf-8 -*-
"""A 2-dim polynomial extension module.

Notes
-----
This extensioin module does not handle errors caused by inconsistent
input parameters.  Validation of input parameters must be done in a
corresponding module.
"""
# cython: cdivision=True
from __future__ import division
cimport cython
cimport numpy as np
from numpy cimport (npy_intp, NPY_DOUBLE, ndarray, PyArray_ZEROS)


np.import_array()

ctypedef np.float64_t DTYPE_t


@cython.wraparound(False)
@cython.boundscheck(False)
def poly1(ndarray[DTYPE_t, ndim=1] xi,
          ndarray[DTYPE_t, ndim=1] yi,
          ndarray[DTYPE_t, ndim=1] cx,
          ndarray[DTYPE_t, ndim=1] cy):
    cdef:
        Py_ssize_t n = xi.shape[0]
        ndarray[DTYPE_t, ndim=1] xo = \
                PyArray_ZEROS(1, <npy_intp*>&n, NPY_DOUBLE, 0)
        ndarray[DTYPE_t, ndim=1] yo = \
                PyArray_ZEROS(1, <npy_intp*>&n, NPY_DOUBLE, 0)

    cdef Py_ssize_t i

    for i in range(n):
        xo[i] = cx[0] + cx[1]*xi[i] + cx[2]*yi[i]
        yo[i] = cy[0] + cy[1]*xi[i] + cy[2]*yi[i]

    return xo, yo


@cython.wraparound(False)
@cython.boundscheck(False)
def poly2(ndarray[DTYPE_t, ndim=1] xi,
          ndarray[DTYPE_t, ndim=1] yi,
          ndarray[DTYPE_t, ndim=1] cx,
          ndarray[DTYPE_t, ndim=1] cy):
    cdef:
        Py_ssize_t n = xi.shape[0]
        ndarray[DTYPE_t, ndim=1] xo = \
                PyArray_ZEROS(1, <npy_intp*>&n, NPY_DOUBLE, 0)
        ndarray[DTYPE_t, ndim=1] yo = \
                PyArray_ZEROS(1, <npy_intp*>&n, NPY_DOUBLE, 0)

    cdef:
        double x, y
        Py_ssize_t i

    for i in range(n):
        x = xi[i]
        y = yi[i]

        xo[i] = cx[0] + cx[1]*x + cx[2]*y + \
                cx[3]*x*x + cx[4]*x*y + cx[5]*y*y
        yo[i] = cy[0] + cy[1]*x + cy[2]*y + \
                cy[3]*x*x + cy[4]*x*y + cy[5]*y*y

    return xo, yo


@cython.wraparound(False)
@cython.boundscheck(False)
def poly3(ndarray[DTYPE_t, ndim=1] xi,
          ndarray[DTYPE_t, ndim=1] yi,
          ndarray[DTYPE_t, ndim=1] cx,
          ndarray[DTYPE_t, ndim=1] cy):
    cdef:
        Py_ssize_t n = xi.shape[0]
        ndarray[DTYPE_t, ndim=1] xo = \
                PyArray_ZEROS(1, <npy_intp*>&n, NPY_DOUBLE, 0)
        ndarray[DTYPE_t, ndim=1] yo = \
                PyArray_ZEROS(1, <npy_intp*>&n, NPY_DOUBLE, 0)

    cdef:
        double x, x2, x3
        double y, y2, y3
        Py_ssize_t i

    for i in range(n):
        x = xi[i]; x2 = x*x; x3 = x2*x
        y = yi[i]; y2 = y*y; y3 = y2*y

        xo[i] = cx[0] + cx[1]*x + cx[2]*y + \
                cx[3]*x2 + cx[4]*x*y + cx[5]*y2 + \
                cx[6]*x3 + cx[7]*x2*y + cx[8]*x*y2 + cx[9]*y3
        yo[i] = cy[0] + cy[1]*x + cy[2]*y + \
                cy[3]*x2 + cy[4]*x*y + cy[5]*y2 + \
                cy[6]*x3 + cy[7]*x2*y + cy[8]*x*y2 + cy[9]*y3

    return xo, yo


@cython.wraparound(False)
@cython.boundscheck(False)
def poly4(ndarray[DTYPE_t, ndim=1] xi,
          ndarray[DTYPE_t, ndim=1] yi,
          ndarray[DTYPE_t, ndim=1] cx,
          ndarray[DTYPE_t, ndim=1] cy):
    cdef:
        Py_ssize_t n = xi.shape[0]
        ndarray[DTYPE_t, ndim=1] xo = \
                PyArray_ZEROS(1, <npy_intp*>&n, NPY_DOUBLE, 0)
        ndarray[DTYPE_t, ndim=1] yo = \
                PyArray_ZEROS(1, <npy_intp*>&n, NPY_DOUBLE, 0)

    cdef:
        double x, x2, x3, x4
        double y, y2, y3, y4
        Py_ssize_t i

    for i in range(n):
        x = xi[i]; x2 = x*x; x3 = x2*x; x4 = x2*x2
        y = yi[i]; y2 = y*y; y3 = y2*y; y4 = y2*y2

        xo[i] = cx[0] + cx[1]*x + cx[2]*y + \
                cx[3]*x2 + cx[4]*x*y + cx[5]*y2 + \
                cx[6]*x3 + cx[7]*x2*y + cx[8]*x*y2 + cx[9]*y3 + \
                cx[10]*x4 + cx[11]*x3*y + cx[12]*x2*y2 + cx[13]*x*y3 + cx[14]*y4
        yo[i] = cy[0] + cy[1]*x + cy[2]*y + \
                cy[3]*x2 + cy[4]*x*y + cy[5]*y2 + \
                cy[6]*x3 + cy[7]*x2*y + cy[8]*x*y2 + cy[9]*y3 + \
                cy[10]*x4 + cy[11]*x3*y + cy[12]*x2*y2 + cy[13]*x*y3 + cy[14]*y4

    return xo, yo
