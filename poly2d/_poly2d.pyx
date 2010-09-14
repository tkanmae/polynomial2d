# -*- coding: utf-8 -*-
"""A 2-dim polynomial extension module.

Notes
-----
This extensioin module does not handle errors caused by inconsistent
input parameters.  Validation of input parameters must be done in
modules which call this extension module.
"""
# cython: cdivision=True
from __future__ import division
cimport cython
from numpy cimport *


## Initialize numpy.
import_array()


@cython.wraparound(False)
@cython.boundscheck(False)
def poly1(ndarray[float64_t,ndim=1] x,
          ndarray[float64_t,ndim=1] y,
          ndarray[float64_t,ndim=1] c):
    cdef:
        Py_ssize_t n = x.shape[0]
        ndarray[float64_t,ndim=1] z = \
                PyArray_ZEROS(1, <npy_intp*>&n, NPY_DOUBLE, 0)
        Py_ssize_t i

    for i in range(n):
        z[i] = c[0] + c[1]*x[i] + c[2]*y[i]
    return z


@cython.wraparound(False)
@cython.boundscheck(False)
def poly2(ndarray[float64_t,ndim=1] x,
          ndarray[float64_t,ndim=1] y,
          ndarray[float64_t,ndim=1] c):
    cdef:
        Py_ssize_t n = x.shape[0]
        ndarray[float64_t,ndim=1] z = \
                PyArray_ZEROS(1, <npy_intp*>&n, NPY_DOUBLE, 0)
        Py_ssize_t i
        double x1, y1

    for i in range(n):
        x1 = x[i]
        y1 = y[i]
        z[i] = c[0] + c[1]*x1 + c[2]*y1 + c[3]*x1*x1 + c[4]*x1*y1 + c[5]*y1*y1
    return z


@cython.wraparound(False)
@cython.boundscheck(False)
def poly3(ndarray[float64_t,ndim=1] x,
          ndarray[float64_t,ndim=1] y,
          ndarray[float64_t,ndim=1] c):
    cdef:
        Py_ssize_t n = x.shape[0]
        ndarray[float64_t,ndim=1] z = \
                PyArray_ZEROS(1, <npy_intp*>&n, NPY_DOUBLE, 0)
        Py_ssize_t i
        double x1, x2, x3
        double y1, y2, y3

    for i in range(n):
        x1 = x[i]; x2 = x1*x1; x3 = x2*x1
        y1 = y[i]; y2 = y1*y1; y3 = y2*y1

        z[i] = c[0] + c[1]*x1 + c[2]*y1 + \
               c[3]*x2 + c[4]*x1*y1 + c[5]*y2 + \
               c[6]*x3 + c[7]*x2*y1 + c[8]*x1*y2 + c[9]*y3
    return z


@cython.wraparound(False)
@cython.boundscheck(False)
def poly4(ndarray[float64_t,ndim=1] x,
          ndarray[float64_t,ndim=1] y,
          ndarray[float64_t,ndim=1] c):
    cdef:
        Py_ssize_t n = x.shape[0]
        ndarray[float64_t,ndim=1] z = \
                PyArray_ZEROS(1, <npy_intp*>&n, NPY_DOUBLE, 0)
        Py_ssize_t i
        double x1, x2, x3, x4
        double y1, y2, y3, y4

    for i in range(n):
        x1 = x[i]; x2 = x1*x1; x3 = x2*x1; x4 = x2*x2
        y1 = y[i]; y2 = y1*y1; y3 = y2*y1; y4 = y2*y2

        z[i] = c[0] + c[1]*x1 + c[2]*y1 + \
               c[3]*x2 + c[4]*x1*y1 + c[5]*y2 + \
               c[6]*x3 + c[7]*x2*y1 + c[8]*x1*y2 + c[9]*y3 + \
               c[10]*x4 + c[11]*x3*y1 + c[12]*x2*y2 + c[13]*x1*y3 + c[14]*y4
    return z


@cython.wraparound(False)
@cython.boundscheck(False)
def poly5(ndarray[float64_t,ndim=1] x,
          ndarray[float64_t,ndim=1] y,
          ndarray[float64_t,ndim=1] c):
    cdef:
        Py_ssize_t n = x.shape[0]
        ndarray[float64_t,ndim=1] z = \
                PyArray_ZEROS(1, <npy_intp*>&n, NPY_DOUBLE, 0)
        Py_ssize_t i
        double x1, x2, x3, x4, x5
        double y1, y2, y3, y4, y5

    for i in range(n):
        x1 = x[i]; x2 = x1*x1; x3 = x2*x1; x4 = x2*x2; x5 = x2*x3
        y1 = y[i]; y2 = y1*y1; y3 = y2*y1; y4 = y2*y2; y5 = y2*y3

        z[i] = c[0] + c[1]*x1 + c[2]*y1 + \
               c[3]*x2 + c[4]*x1*y1 + c[5]*y2 + \
               c[6]*x3 + c[7]*x2*y1 + c[8]*x1*y2 + c[9]*y3 + \
               c[10]*x4 + c[11]*x3*y1 + c[12]*x2*y2 + c[13]*x1*y3 + c[14]*y4 + \
               c[15]*x5 + c[16]*x4*y1 + c[17]*x3*y2 + c[18]*x2*y3 + c[19]*x1*y4 + c[20]*y5
    return z

