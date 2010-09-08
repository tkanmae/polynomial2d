#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import *

from poly2d import (poly2d, polyfit2d, poly2d_transform, polyfit2d_transform)


class TestPoly2d(TestCase):

    def test_call(self):
        c = [1, 2, 3]
        p = poly2d(c)

        ## x and y must have the same size.
        x = np.random.random(10)
        y = np.random.random(11)
        assert_raises(ValueError, p, x, y)


class TestPoly2dTransform(TestCase):

    def test_constructor(self):
        ## cx and cy must have the same size.
        cx = [1, 2, 3]
        cy = [1, 2, 3, 4]
        assert_raises(ValueError, poly2d_transform, cx, cy)

    def test_call(self):
        cx = [1, 2, 3]
        cy = [1, 2, 3]
        p = poly2d_transform(cx, cy)

        ## x and y must have the same size.
        x = np.random.random(10)
        y = np.random.random(11)
        assert_raises(ValueError, p, x, y)


class TestPolyfit2d(TestCase):

    def test_input_size(self):
        x  = np.random.random(10)
        y  = np.random.random(10)
        z  = np.random.random(10)

        ## The input must have the same size.
        e = np.random.random(11)
        assert_raises(ValueError, polyfit2d, x, y, e)
        assert_raises(ValueError, polyfit2d, x, e, z)
        assert_raises(ValueError, polyfit2d, e, z, z)


class TestPolyfit2dTransform(TestCase):

    def test_input_size(self):
        x  = np.random.random(10)
        y  = np.random.random(10)
        xt = np.random.random(10)
        yt = np.random.random(10)

        ## The input must have the same size.
        e = np.random.random(11)
        assert_raises(ValueError, polyfit2d_transform, x, y, e, yt)
        assert_raises(ValueError, polyfit2d_transform, x, y, xt, e)
        assert_raises(ValueError, polyfit2d_transform, x, e, xt, yt)
        assert_raises(ValueError, polyfit2d_transform, e, y, xt, yt)


if __name__ == '__main__':
    run_module_suite()
