polynomial2d
============

## Features

`polynomial2d` provides `polyfit2d()`, a 2-dim polynomial fitting
routine. It determines the coefficients of a 2-dim polynomial of a given
degree `n`, using the least square fit to given data values `z` at given
points `(x, y)`.  The function assumes the polynomial in a form of

    p(x,y) = \\sum_{i,j} c_{i,j} * x^i * y^j

with a constraint of `i + j <= n`.  The coefficients are returned in a
2-dim array.  That allows them to work with relevant polynomial
functions in `numpy.polynomial.polynomial`.

For consistency, `polynomial2d` also provides `polyval2d()`, a wrapper
around `polyval2d()` in `numpy.polynomial.polynomial`.


## Examples

```python
>>> import numpy as np
>>> from polynomial2d import polyfit2d

>>> x, y = np.random.rand(20), np.random.rand(20)
>>> z = 2.0 + 3.0 * x - y + 0.1 * x**2 - 0.01 * x * y - 0.2 * y**2
>>> # Assume a 2-dim polynomial of degree 2.
>>> c = polyfit2d(x, y, z, deg=2)
>>> print(c)
[[ 2.   -1.   -0.2 ]
 [ 3.   -0.01  0.  ]
 [ 0.1   0.    0.  ]]
```

Note that the coefficients are returned in an array with a shape of (3,
3).  The form of the coefficients nicely play with 2-dim polynomial
functions in `numpy.polynomial.polynomial`.

```python
>>> import numpy.polynomial.polynomial import polyval2d
>>> np.allclose(polyval2d(x, y, c), z)
True
```


## Thanks

Most of code in the polynomial fitting routine were borrowed from
`polyfit()` in `numpy.polynomial.polynomial`.

