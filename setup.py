#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-
import os

pjoin = os.path.join

root_dir = 'polynomial2d'


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('polynomial2d',
                           parent_package,
                           top_path,
                           package_path=root_dir)

    ## Add `_poly2d` extension module.
    src_files = ['_poly2d.c']
    src_files = [pjoin(root_dir, f) for f in src_files]
    config.add_extension('_poly2d', sources=src_files)

    ## Add `tests` directory.
    config.add_data_dir(('tests', pjoin(root_dir, 'tests')))

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(
          version       = '0.1.0',
          author        = 'Takeshi Kaname',
          author_email  = 'tkanmae@gmail.com',
          keywords      = ['numpy', 'data', 'science'],
          configuration = configuration,
        )
