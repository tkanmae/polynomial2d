#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from numpy.testing import *

from poly2d import (poly2d, polyfit2d, poly2d_spatial, polyfit2d_spatial)


def _realistic_coeffs(order):
    """Return realistic polynomial coefficients."""
    from random import random
    # -- Pairs of the magnitude and the number of coefficients.
    mags = ((100*random(), 1), (100, 2), (10, 3), (1, 4), (0.1, 5))
    if order < 1:
        raise ValueError('`order` must be larger than 1.')
    ret = []
    for i, (mag, n) in enumerate(mags):
        ret.append([mag*random() for j in range(n)])
        if i == order: break
    return np.hstack(ret)


class TestPoly2d(TestCase):

    def _run(self, c0, order):
        x = 100 * np.random.random(1000)
        y = 100 * np.random.random(1000)

        z = poly2d(c0)(x, y)
        c = polyfit2d(x, y, z, order)

        assert_array_almost_equal(c, c0)

    def test_polyorder2(self):
        for i in range(10):
            self._run(_realistic_coeffs(2), order=2)

    def test_polyorder3(self):
        for i in range(10):
            self._run(_realistic_coeffs(3), order=3)

    def test_polyorder4(self):
        for i in range(10):
            self._run(_realistic_coeffs(4), order=4)


class TestPoly2dSpatial(TestCase):

    def _run(self, cx0, cy0, order):
        x = 100 * np.random.random(1000)
        y = 100 * np.random.random(1000)

        xt, yt = poly2d_spatial(cx0, cy0)(x, y)
        cx, cy = polyfit2d_spatial(x, y, xt, yt, order)

        assert_array_almost_equal(cx, cx0)
        assert_array_almost_equal(cy, cy0)

    def test_polyorder2(self):
        for i in range(10):
            self._run(_realistic_coeffs(2), _realistic_coeffs(2), order=2)

    def test_polyorder3(self):
        for i in range(10):
            self._run(_realistic_coeffs(3), _realistic_coeffs(3), order=3)

    def test_polyorder4(self):
        for i in range(10):
            self._run(_realistic_coeffs(4), _realistic_coeffs(4), order=4)

    def test_affine_scale(self):
        cx = [0, 3, 0]
        cy = [0, 0, 1/3]
        self._run(cx, cy, order=1)

    def test_affine_translation(self):
        cx = [10, 0, 0]
        cy = [-7, 0, 0]
        self._run(cx, cy, order=1)

    def test_affine_rotation(self):
        from math import (cos, sin, radians)

        theta = 60
        sin_theta = sin(radians(theta))
        cos_theta = cos(radians(theta))
        cx = [0,  cos_theta, sin_theta]
        cy = [0, -sin_theta, cos_theta]
        self._run(cx, cy, order=1)


if __name__ == '__main__':
    run_module_suite()
