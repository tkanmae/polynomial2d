# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import *

from poly2d import poly2d, polyfit2d


class TestPoly2d(TestCase):

    def test_different_size_coeffs(self):
        cx = [1, 2, 3]
        cy = [1, 2, 3, 4]
        self.failUnlessRaises(ValueError, poly2d, cx, cy)

    def test_invalid_ncoeffs(self):
        valid_ncoeffs = set((3, 6, 10, 15))
        invalid_ncoeffs = set(range(100)).difference(valid_ncoeffs)
        for n in invalid_ncoeffs:
            cx = np.random.random(n)
            cy = np.random.random(n)
            self.failUnlessRaises(ValueError, poly2d, cx, cy)

    def test_different_size_xy(self):
        cx = [1, 2, 3]
        cy = [1, 2, 3]
        p = poly2d(cx, cy)
        x = np.random.random(10)
        y = np.random.random(11)
        self.failUnlessRaises(ValueError, p, x, y)

class TestPolyfit2d(TestCase):

    def test_different_size_xy(self):
        x  = np.random.random(10)
        y  = np.random.random(10)
        xt = np.random.random(10)
        yt = np.random.random(10)

        z = np.random.random(11)

        self.failUnlessRaises(ValueError,
                              polyfit2d, x, z, xt, yt)
        self.failUnlessRaises(ValueError,
                              polyfit2d, x, y, z, yt)
        self.failUnlessRaises(ValueError,
                              polyfit2d, x, y, xt, z)

if __name__ == '__main__':
    run_module_suite()
