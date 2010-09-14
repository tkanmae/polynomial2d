#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
import numpy as np
from functools import partial
try:
    import unittest2 as unittest
except:
    import unittest
from numpy.testing import *

from poly2d.polynomial import (ncoeffs, order, vandermonde, )
from poly2d.polynomial import (poly2d, polyfit2d, poly2d_transform, polyfit2d_transform,)


def _realistic_coeffs(order):
    """Return realistic polynomial coefficients."""
    from random import random
    ## Pairs of the magnitude and the number of coefficients.
    mags = ((100*random(), 1), (100, 2), (10, 3), (1, 4), (0.1, 5), (0.01, 6))
    if order < 1:
        raise ValueError('`order` must be larger than 1.')
    ret = []
    for i, (mag, n) in enumerate(mags):
        ret.append([mag*random() for j in range(n)])
        if i == order: break
    return np.hstack(ret)


class TestHelperFuncs(unittest.TestCase):

    def test_ncoeffs(self):
        self.assertEqual(3, ncoeffs(1))
        self.assertEqual(6, ncoeffs(2))
        self.assertEqual(10, ncoeffs(3))
        self.assertEqual(15, ncoeffs(4))
        self.assertEqual(21, ncoeffs(5))

        ## `order` must be a positive integer.
        self.assertRaises(TypeError, ncoeffs, 1.0)
        self.assertRaises(ValueError, ncoeffs, 0)
        self.assertRaises(ValueError, ncoeffs, -1)

    def test_order(self):
        self.assertEqual(1, order(3))
        self.assertEqual(2, order(6))
        self.assertEqual(3, order(10))
        self.assertEqual(4, order(15))
        self.assertEqual(5, order(21))

        ## `ncoeffs` must be one of 3, 6, 10, ...
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


class TestPoly2d(unittest.TestCase):

    def test_ctor(self):
        self.assertTrue(isinstance(poly2d([1, 2, 3]), poly2d))
        self.assertTrue(isinstance(poly2d([1, 2, 3, 4, 5, 6]), poly2d))

        ## `coeffs` must be a 1-dim array.
        coeffs = [[1, 2, 3], [4, 5, 6]]
        self.assertRaises(ValueError, poly2d, coeffs)
        ## `coeffs` must be consistent size.
        coeffs = [[1, 2, 3, 4], [5, 6, 7, 9]]
        self.assertRaises(ValueError, poly2d, coeffs)

    def test_attr(self):
        p = poly2d([1, 2, 3])

        self.assertTrue(hasattr(p, 'coeffs'))
        self.assertTrue(hasattr(p, 'order'))

        ## __len__() returns the polynomial order.
        self.assertEqual(1, len(p))
        ## __array__() returns a copy of `coeffs`.
        c = np.array(p)
        assert_equal(p, p.coeffs)
        self.assertNotEqual(id(c), id(p.coeffs))

    def test_call(self):
        p = poly2d([1, 2, 3])

        assert_equal(p(0, 0), 1)
        assert_equal(p(1, 2), 9)

        ## x and y must be a 1-dim array.
        x = np.random.random(10).reshape(2,5)
        y = np.random.random(10).reshape(2,5)
        self.assertRaises(ValueError, p, x, y)
        ## x and y must have the same size.
        x = np.random.random(10)
        y = np.random.random(11)
        self.assertRaises(ValueError, p, x, y)


class TestPoly2dTransform(unittest.TestCase):

    def test_ctor(self):
        self.assertTrue(
            isinstance(poly2d_transform([1, 2, 3], [1, 2, 3]), poly2d_transform))
        ## `coeffs` must be a 1-dim array.
        c = [[1, 2, 3], [4, 5, 6]]
        self.assertRaises(ValueError, poly2d_transform, c, c)
        ## `coeffs` must be consistent size.
        c = [[1, 2, 3, 4], [5, 6, 7, 9]]
        self.assertRaises(ValueError, poly2d_transform, c, c)
        ## cx and cy must have the same size.
        self.assertRaises(ValueError, poly2d_transform, [1, 2, 3], [1, 2, 3, 4])

    def test_attr(self):
        p = poly2d_transform([1, 2, 3], [1, 2, 3])

        self.assertTrue(hasattr(p, 'cx'))
        self.assertTrue(hasattr(p, 'cy'))
        self.assertTrue(hasattr(p, 'order'))

    def test_call(self):
        p = poly2d_transform([1, 2, 3], [1, 2, 3])

        assert_equal(p(0, 0), (1, 1))
        assert_equal(p(1, 2), (9, 9))

        ## x and y must be a 1-dim array.
        x = np.random.random(10).reshape(2,5)
        y = np.random.random(10).reshape(2,5)
        self.assertRaises(ValueError, p, x, y)
        ## x and y must have the same size.
        x = np.random.random(10)
        y = np.random.random(11)
        self.assertRaises(ValueError, p, x, y)

    def test_affine_translation(self):
        x1 = np.array([1, 2, 3])
        y1 = np.array([4, 5, 6])
        cx = [1, 1, 0]
        cy = [-1, 0, 1]
        p = poly2d_transform(cx, cy)
        x2, y2 = p(x1, y1)
        assert_equal(x1 + 1, x2)
        assert_equal(y1 - 1, y2)

    def test_affine_scaling(self):
        x1 = np.array([1, 2, 3])
        y1 = np.array([4, 5, 6])
        cx = [0, 2, 0]
        cy = [0, 0, -1]
        p = poly2d_transform(cx, cy)
        x2, y2 = p(x1, y1)
        assert_equal(2*x1, x2)
        assert_equal(-1*y1, y2)

    def test_affine_rotation(self):
        x1 = np.array([1, 2, 3])
        y1 = np.array([0, 0, 0])
        cx, cy = self._rot_coeffs(-90)
        p = poly2d_transform(cx, cy)
        x2, y2 = p(x1, y1)
        assert_almost_equal(y1, x2)
        assert_almost_equal(x1, y2)

        cx, cy = self._rot_coeffs(180)
        p = poly2d_transform(cx, cy)
        x2, y2 = p(x1, y1)
        assert_almost_equal(-x1, x2)
        assert_almost_equal(y1, y2)

    def _rot_coeffs(self, theta):
        from math import (cos, sin, radians)
        sin_theta = sin(radians(theta))
        cos_theta = cos(radians(theta))
        cx = [0,  cos_theta, sin_theta]
        cy = [0, -sin_theta, cos_theta]
        return cx, cy


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

        ## The input must be 1-dim arrays.
        self.assertRaises(ValueError, polyfit2d, x.reshape(2,5), y, z)
        self.assertRaises(ValueError, polyfit2d, x, y.reshape(2,5), z)
        self.assertRaises(ValueError, polyfit2d, x, y, z.reshape(2,5))
        ## The input must have the same size.
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

        z = poly2d(c0)(x, y)
        c = polyfit2d(x, y, z, order)

        assert_array_almost_equal(c, c0)


class TestPolyfit2dTransform(unittest.TestCase):

    def test_interface(self):
        x1 = np.random.random(10)
        y1 = np.random.random(10)
        x2 = np.random.random(10)
        y2 = np.random.random(10)

        rv = polyfit2d_transform(x1, y1, x2, y2)
        self.assertEqual(2, len(rv))
        rv = polyfit2d_transform(x1, y1, x2, y2, full_output=True)
        self.assertEqual(8, len(rv))

        ## The input must be 1-dim arrays.
        self.assertRaises(ValueError, polyfit2d_transform, x1.reshape(2,5), y1, x2, y2)
        self.assertRaises(ValueError, polyfit2d_transform, x1, y1.reshape(2,5), x2, y2)
        self.assertRaises(ValueError, polyfit2d_transform, x1, y1, x2.reshape(2,5), y2)
        self.assertRaises(ValueError, polyfit2d_transform, x1, y1, x2, y2.reshape(2,5))
        ## The input must have the same size.
        e = np.random.random(11)
        assert_raises(ValueError, polyfit2d_transform, e, y1, x2, y2)
        assert_raises(ValueError, polyfit2d_transform, x1, e, x2, y2)
        assert_raises(ValueError, polyfit2d_transform, x1, y1, e, y2)
        assert_raises(ValueError, polyfit2d_transform, x1, y1, x2, e)

    def test_lin(self):
        x = np.random.random(10)
        y = np.random.random(10)

        x2 = x + y
        y2 = x - y
        cx, cy = polyfit2d_transform(x, y, x2, y2)
        assert_almost_equal([0, 1, 1], cx)
        assert_almost_equal([0, 1, -1], cy)

        x2 = -1 - x + y
        y2 =  3 + x - y
        cx, cy = polyfit2d_transform(x, y, x2, y2)
        assert_almost_equal([-1, -1, 1], cx)
        assert_almost_equal([3, 1, -1], cy)

    def test_quad(self):
        P2 = partial(polyfit2d_transform, order=2)
        x = np.random.random(10)
        y = np.random.random(10)

        x2 = x*x + x*y + y*y
        y2 = x*x - x*y - y*y
        cx, cy = P2(x, y, x2, y2)
        assert_almost_equal([0, 0, 0, 1, 1, 1], cx)
        assert_almost_equal([0, 0, 0, 1, -1, -1], cy)

        x2 = -1 + x - y + x*x + x*y + y*y
        y2 =  3 - x + y + x*x - x*y - y*y
        cx, cy = P2(x, y, x2, y2)
        assert_almost_equal([-1, 1, -1, 1, 1, 1], cx)
        assert_almost_equal([3, -1, 1, 1, -1, -1], cy)

    def test_cube(self):
        P3 = partial(polyfit2d_transform, order=3)
        x = np.random.random(20)
        y = np.random.random(20)

        x2 = x**3 + x**2*y + x*y**2 + y**3
        y2 = x**3 - x**2*y + x*y**2 - y**3
        cx, cy = P3(x, y, x2, y2)
        assert_almost_equal([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], cx)
        assert_almost_equal([0, 0, 0, 0, 0, 0, 1, -1, 1, -1], cy)

        x2 = -1 + x - y + x*x + x*y + y*y + x**3 + x**2*y - x*y**2 + y**3
        y2 =  3 - x + y + x*x - x*y - y*y + x**3 - x**2*y + x*y**2 - y**3
        cx, cy = P3(x, y, x2, y2)
        assert_almost_equal([-1, 1, -1, 1, 1, 1, 1, 1, -1, 1], cx)
        assert_almost_equal([3, -1, 1, 1, -1, -1, 1, -1, 1, -1], cy)

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
        cx0 = _realistic_coeffs(order)
        cy0 = _realistic_coeffs(order)
        x = 100 * np.random.random(1000)
        y = 100 * np.random.random(1000)

        x2, y2 = poly2d_transform(cx0, cy0)(x, y)
        cx, cy = polyfit2d_transform(x, y, x2, y2, order)

        assert_array_almost_equal(cx, cx0)
        assert_array_almost_equal(cy, cy0)
