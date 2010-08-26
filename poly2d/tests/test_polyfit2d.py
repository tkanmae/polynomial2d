#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
from __future__ import division
from math import (cos, sin, radians)
import numpy as np
from numpy.testing import *

from poly2d import poly2d, polyfit2d


def run(cx0, cy0, order):
    x = 100 * np.random.random(1000)
    y = 100 * np.random.random(1000)
    p = poly2d(cx0, cy0)

    xt, yt = p(x, y)
    cx, cy = polyfit2d(x, y, xt, yt, order)

    assert_array_almost_equal(cx, cx0)
    assert_array_almost_equal(cy, cy0)


def realistic_coeffs(order):
    """Emulate realistic polynomial coefficients."""
    from random import random
    # -- Pairs of the magnitude and the number of coefficients.
    mags = ((100*random(), 1), (100, 2), (10, 3), (1, 4), (0.1, 5))
    if order < 1:
        raise ValueError('`order` must be larger than 1.')
    ret = []
    for i, (mag, n) in enumerate(mags):
        ret.append([random() for i in range(n)])
        if i == order:
            break
    return np.hstack(ret)


class TestAffineTransform(TestCase):

    def setUp(self):
        theta = 60
        self.cos_theta = cos(radians(theta))
        self.sin_theta = sin(radians(theta))

        self.tx = 10
        self.ty = -7

        self.scx = 3
        self.scy = 1 / 3

    def test_scale(self):
        cx = [0, self.scx, 0]
        cy = [0, 0, self.scy]
        run(cx, cy, order=1)

    def test_trans(self):
        cx = [self.tx, 0, 0]
        cy = [self.ty, 0, 0]
        run(cx, cy, order=1)

    def test_rot(self):
        cx = [0,  self.cos_theta, self.sin_theta]
        cy = [0, -self.sin_theta, self.cos_theta]
        run(cx, cy, order=1)

    def test_transform(self):
        for i in range(100):
            cx = realistic_coeffs(1)
            cy = realistic_coeffs(1)
            run(cx, cy, order=1)


class TestPoly2Transform(TestCase):

    def test_transform(self):
        for i in range(10):
            cx = realistic_coeffs(2)
            cy = realistic_coeffs(2)
            run(cx, cy, order=2)


class TestPoly3Transform(TestCase):

    def test_transform(self):
        for i in range(10):
            cx = realistic_coeffs(3)
            cy = realistic_coeffs(3)
            run(cx, cy, order=3)


class TestPoly4Transform(TestCase):

    def test_transform(self):
        for i in range(10):
            cx = realistic_coeffs(4)
            cy = realistic_coeffs(4)
            run(cx, cy, order=4)


if __name__ == '__main__':
    run_module_suite()
