polynomial2d
============

## Features

`polynomial2d` provides `polyfit2d()`, a 2-dim polynomial fitting routine.  It
determines 2-dim polynomial coefficients of a given degree `n`, using a least
square fit to given data values `z` at given points `(x, y)`.  It assumes a
polynomial in a form of

    z(x,y) = \\sum_{i,j} c_{i,j} * x^i * y^j

with a constraint of `i + j <= n`.  The coefficients are returned in a 2-dim
array in a form that allows to work with relevant polynomial functions in
`numpy.polynomial.polynomial`.

`polynomial2d` also provides, for consistency, `polyval2d()` and
`polygrid1d()`, wrapper functions of those in `numpy.polynomial.polynomial`.


## Examples

```python
>>> import numpy as np
>>> from polynomial2d import polyfit2d

>>> x = np.random.rand(20)
>>> y = np.random.rand(20)
>>> z = 2.0 + 3.0 * x - y + 0.1 * x**2 - 0.01 * x * y - 0.2 * y**2
>>> c = polyfit2d(x, y, z, deg=2)
>>> print(c)
[[ 2.   -1.   -0.2 ]
 [ 3.   -0.01  0.  ]
 [ 0.1   0.    0.  ]]
```

Note that the coefficients are returned in an array with a shape of (3, 3).
The coefficients are arranged so that the returned array nicely work with 2-dim
polynomial functions in `numpy.polynomial.polynomial`.

```python
>>> import numpy.polynomial.polynomial import polyval2d
>>> np.allclose(polyval2d(x, y, c), z)
True
```


## Thanks

Most of code in the polynomial fitting routine were borrowed from
`polyfit()` in `numpy.polynomial.polynomial`.

