# -*- coding: utf-8 -*-
from os.path import join

from numpy.distutils.command import build_src
try:
    import Cython
    import Cython.Compiler.Main
    build_src.Pyrex = Cython
    build_src.have_pyrex = True
except ImportError, e:
    print 'You must need Cython installed.'
    raise e

_root_dir = 'poly2d'
_src_dir  = join(_root_dir, 'src')


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('poly2d',
                           parent_package,
                           top_path,
                           package_path =_root_dir)

    # -- Add `_poly2d` extension module.
    src_files = ['_poly2d.pyx']
    src_files = [join(_src_dir, f) for f in src_files]
    config.add_extension('_poly2d',
                         sources=src_files,
                         include_dirs=[_src_dir])

    # -- Add `tests` directory.
    config.add_data_dir(('tests', join(_root_dir, 'tests')))

    return config


if __name__ == '__main__':
    import setuptools
    from numpy.distutils.core import setup

    setup(name          = 'poly2d',
          version       = '0.1.0',
          author        = 'Takeshi Kaname',
          author_email  = 'tkanmae@gmail.com',
          keywords      = ['numpy', 'data', 'science'],
          configuration = configuration,
          test_suite    = 'nose.collector',
        )
