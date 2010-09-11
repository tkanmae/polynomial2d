======
poly2d
======

A tiny library for 2d polynomial.


Installation
============

Building poly2d requires the following software installed:

* Python (>=2.6)
* NumPy (>=1.3)
* [optional] nose (>=0.11)

nose is required to execute tests.

In order to build poly2d, simply do::

    $ python setup.py build
    $ python setup.py install

Then, verify a successful installation::

    $ python -c "import poly2d; poly2d.test()"


If you downloaded poly2d from the GitHub repository, you need to have
Cython (>=0.13) installed.

::

    $ cython -v poly2d/_poly2d.pyx
    $ python setup.py build
    $ python setup.py install
    $ python -c "import poly2d; poly2d.test()"

If you just want to try poly2d without installing it, build it
in-place::

    $ (cython -v poly2d/_poly2d.pyx)
    $ python setup.py build_ext --inplace -f
    $ python -c "import poly2d; poly2d.test()"


Authors
=======

Takeshi Kanmae <tkanmae@gmail.com>


License
=======

The MIT license applies to the software.  See the file LICENSE.txt.


Resources
=========

* Python: http://www.python.org/
* NumPy: http://www.scipy.org/
* nose: http://somethingaboutorange.com/mrl/projects/nose
* Cython: http://www.cython.org/


.. # vim: ft=rst tw=72
