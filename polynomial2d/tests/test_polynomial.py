from __future__ import absolute_import, division, print_function

import numpy as np
from nose.tools import assert_true
from numpy.testing import (assert_almost_equal, assert_equal, assert_raises)

from ..polynomial import _polyvander2d
from ..polynomial import (polyval2d, polyfit2d,)


def test_polyvander2d():
    x, y = np.array([1]), np.array([2])

    desired = np.array([[1, 1, 2]])
    assert_equal(_polyvander2d(x, y, 1), desired)
    desired = np.array([[1, 1, 2, 1, 2, 4]])
    assert_equal(_polyvander2d(x, y, 2), desired)
    desired = np.array([[1, 1, 2, 1, 2, 4, 1, 2, 4, 8]])
    assert_equal(_polyvander2d(x, y, 3), desired)
    desired = np.array([[1, 1, 2, 1, 2, 4, 1, 2, 4, 8, 1, 2, 4, 8, 16]])
    assert_equal(_polyvander2d(x, y, 4), desired)


def test_polyval2d():
    # Coeffcients must be in a 2-dim array.
    assert_raises(ValueError, polyval2d, 0, 0, [1, 2, 3])
    assert_raises(ValueError, polyval2d, 0, 0, [[1, 2], [3, 4], [4, 5]])

    c = [[1, 2], [3, 0]]
    assert_equal(polyval2d(0, 0, c), 1.0)
    assert_equal(polyval2d(1, 0, c), 4.0)
    assert_equal(polyval2d(0, 1, c), 3.0)
    assert_equal(polyval2d(1, 1, c), 6.0)


def test_polyfit2d():
    # Input arrays must be 1-dim.
    assert_raises(ValueError, polyfit2d, [[1, 2], [3, 4]], [1, 2], [1, 2])
    assert_raises(ValueError, polyfit2d, [1, 2], [[1, 2], [3, 4]], [1, 2])
    assert_raises(ValueError, polyfit2d, [1, 2], [1, 2], [[1, 2], [3, 4]])
    # Input arrays must have the same size.
    assert_raises(ValueError, polyfit2d, [1, 2], [1, 2], [1, 2, 3])
    assert_raises(ValueError, polyfit2d, [1, 2], [1, 2, 3], [1, 2])
    assert_raises(ValueError, polyfit2d, [1, 2], [1, 2], [1, 2, 3])

    x, y = np.random.rand(20), np.random.rand(20)

    z = np.random.rand(20)
    assert_true(len(polyfit2d(x, y, z)), 3)
    assert_true(len(polyfit2d(x, y, z, full_output=True)), 5)

    z = 2 * x - y + 3
    desired = [[3, -1], [2, 0]]
    assert_almost_equal(polyfit2d(x, y, z), np.array(desired))

    z = 1 - x + 2*y + x*x - x*y + y*y
    desired = [[1, 2, 1], [-1, -1, 0], [1, 0, 0]]
    assert_almost_equal(polyfit2d(x, y, z, deg=2), np.array(desired))

    z = 1 + x - y + x*x + x*y + y*y - x**3 + x**2*y - x*y**2 + y**3
    desired = [[1, -1, 1, 1], [1, 1, -1, 0], [1, 1, 0, 0], [-1, 0, 0, 0]]
    assert_almost_equal(polyfit2d(x, y, z, deg=3), np.array(desired))
