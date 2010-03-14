#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
import os

# ----------------------------------------------------------------------
# [2010-01-10]
# This hack does not work with `numpy.distutils` with the version 1.4.0.
# You must execute `cython *.pyx` in src directory and generate C source
# files manually before execute this script.
# ----------------------------------------------------------------------
# -- http://www.mail-archive.com/numpy-discussion@scipy.org/msg19933.html
# from numpy.distutils.command import build_src
# try:
#     import Cython
#     import Cython.Compiler.Main
#     build_src.Pyrex = Cython
#     build_src.have_pyrex = True
# except ImportError, err:
#     print 'You must need Cython installed.'
#     raise err


root_dir = 'poly2d'
src_dir  = os.path.join(root_dir, 'src')

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('poly2d',
                           parent_package,
                           top_path,
                           package_path =root_dir)

    # -- Add `_poly2d` extension module.
    # src_files = ['_poly2d.pyx']
    src_files = ['_poly2d.c']
    src_files = [os.path.join(src_dir, f) for f in src_files]
    config.add_extension('_poly2d',
                         sources=src_files,
                         include_dirs=[src_dir])

    # -- Add `tests` directory.
    config.add_data_dir(('tests', os.path.join(root_dir, 'tests')))

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(
          version       = '0.1.0',
          author        = 'Takeshi Kaname',
          author_email  = 'tkanmae@gmail.com',
          keywords      = ['numpy', 'data', 'science'],
          configuration = configuration,
          test_suite    = 'nose.collector',
        )
