#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
from functools import partial
try:
    import unittest2 as unittest
except:
    import unittest
from numpy.testing import *

from ..polynomial import (ncoeffs, order, vandermonde, )
from ..polynomial import (polyval2d, polyfit2d,)


def _realistic_coeffs(order):
    """Return realistic polynomial coefficients."""
    from random import random
    # Pairs of the magnitude and the number of coefficients.
    mags = ((100*random(), 1), (100, 2), (10, 3), (1, 4), (0.1, 5), (0.01, 6))
    if order < 1:
        raise ValueError('`order` must be larger than 1.')
    ret = []
    for i, (mag, n) in enumerate(mags):
        ret.append([mag*random() for j in range(n)])
        if i == order:
            break
    return np.hstack(ret)


class TestHelperFuncs(unittest.TestCase):

    def test_ncoeffs(self):
        self.assertEqual(3, ncoeffs(1))
        self.assertEqual(6, ncoeffs(2))
        self.assertEqual(10, ncoeffs(3))
        self.assertEqual(15, ncoeffs(4))
        self.assertEqual(21, ncoeffs(5))

        # `order` must be a positive integer.
        self.assertRaises(TypeError, ncoeffs, 1.0)
        self.assertRaises(ValueError, ncoeffs, 0)
        self.assertRaises(ValueError, ncoeffs, -1)

    def test_order(self):
        self.assertEqual(1, order(3))
        self.assertEqual(2, order(6))
        self.assertEqual(3, order(10))
        self.assertEqual(4, order(15))
        self.assertEqual(5, order(21))

        # `ncoeffs` must be one of 3, 6, 10, ...
        self.assertRaises(ValueError, order, 1)
        self.assertRaises(ValueError, order, 4)

    def test_vandermonde(self):
        x = np.array([1])
        y = np.array([2])

        e = np.array([[1, 1, 2]])
        assert_equal(e, vandermonde(x, y, 1))
        e = np.array([[1, 1, 2, 1, 2, 4]])
        assert_equal(e, vandermonde(x, y, 2))
        e = np.array([[1, 1, 2, 1, 2, 4, 1, 2, 4, 8]])
        assert_equal(e, vandermonde(x, y, 3))
        e = np.array([[1, 1, 2, 1, 2, 4, 1, 2, 4, 8, 1, 2, 4, 8, 16]])
        assert_equal(e, vandermonde(x, y, 4))

        x = np.arange(10)
        y = np.arange(10)
        self.assertRaises(ValueError, vandermonde, x.reshape(5,2), y, 1)
        self.assertRaises(ValueError, vandermonde, x, y.reshape(5,2), 1)


def test_polyval2d():
    c = [[1, 2], [3, 0]]
    assert_equal(polyval2d(0, 0, c), 1.0)
    assert_equal(polyval2d(1, 0, c), 4.0)
    assert_equal(polyval2d(0, 1, c), 3.0)
    assert_equal(polyval2d(1, 1, c), 6.0)

    assert_raises(ValueError, polyval2d, 0, 0, [1, 2, 3])
    assert_raises(ValueError, polyval2d, 0, 0, [[1, 2, 3], [1, 2, 3]])


class TestPolyfit2d(unittest.TestCase):

    def test_interface(self):
        x = np.random.random(10)
        y = np.random.random(10)
        z = np.random.random(10)

        rv = polyfit2d(x, y, z)
        self.assertEqual(3, len(rv))
        rv = polyfit2d(x, y, z, full_output=True)
        self.assertEqual(5, len(rv))
        rv = polyfit2d(x, y, z, order=2)
        self.assertEqual(6, len(rv))
        rv = polyfit2d(x, y, z, order=3)
        self.assertEqual(10, len(rv))

        # The input must be 1-dim arrays.
        self.assertRaises(ValueError, polyfit2d, x.reshape(2,5), y, z)
        self.assertRaises(ValueError, polyfit2d, x, y.reshape(2,5), z)
        self.assertRaises(ValueError, polyfit2d, x, y, z.reshape(2,5))
        # The input must have the same size.
        e = np.random.random(11)
        self.assertRaises(ValueError, polyfit2d, x, y, e)
        self.assertRaises(ValueError, polyfit2d, x, e, z)
        self.assertRaises(ValueError, polyfit2d, e, z, z)

    def test_lin(self):
        x = np.random.random(10)
        y = np.random.random(10)

        z = x + y
        assert_almost_equal([0, 1, 1], polyfit2d(x, y, z))
        z = x + y + 3
        assert_almost_equal([3, 1, 1], polyfit2d(x, y, z))
        z = 2*x - y + 3
        assert_almost_equal([3, 2, -1], polyfit2d(x, y, z))

    def test_quad(self):
        P2 = partial(polyfit2d, order=2)
        x = np.random.random(10)
        y = np.random.random(10)

        z = x*x + x*y + y*y
        assert_almost_equal([0, 0, 0, 1, 1, 1], P2(x, y, z))
        z = 1 + x + y + x*x + x*y + y*y
        assert_almost_equal([1, 1, 1, 1, 1, 1], P2(x, y, z))
        z = 1 - x + y + x*x - x*y + y*y
        assert_almost_equal([1, -1, 1, 1, -1, 1], P2(x, y, z))

    def test_cube(self):
        P3 = partial(polyfit2d, order=3)
        x = np.random.random(20)
        y = np.random.random(20)

        z = x**3 + x**2*y + x*y**2 + y**3
        assert_almost_equal(
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1], P3(x, y, z))
        z = 1 + x + y + x*x + x*y + y*y + x**3 + x**2*y + x*y**2 + y**3
        assert_almost_equal(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], P3(x, y, z))
        z = 1 + x - y + x*x + x*y + y*y - x**3 + x**2*y - x*y**2 + y**3
        assert_almost_equal(
            [1, 1, -1, 1, 1, 1, -1, 1, -1, 1], P3(x, y, z))

    def test_rnd_order2(self):
        for i in range(10):
            self._run_rnd(2)

    def test_rnd_order3(self):
        for i in range(10):
            self._run_rnd(3)

    def test_rnd_order4(self):
        for i in range(10):
            self._run_rnd(4)

    def test_rnd_order5(self):
        for i in range(10):
            self._run_rnd(5)

    def _run_rnd(self, order):
        c0 = _realistic_coeffs(order)
        x = 100 * np.random.random(1000)
        y = 100 * np.random.random(1000)

        z = polyval2d(c0)(x, y)
        c = polyfit2d(x, y, z, order)

        assert_array_almost_equal(c, c0)
