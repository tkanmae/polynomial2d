#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from setuptools import setup, Extension


setup(
    name='polynomial2d',
    version='0.2.0',
    license='MIT',
    author='Takeshi Kanmae',
    author_email='tkanmae@gmail.com',
    classifiers=[
        'Intentended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programing Language :: Python',
        'Licence :: OSI Approved :: MIT License',
    ],
    packages=[
        'polynomial2d',
        'polynomial2d.tests',
    ],
    ext_modules=[
        Extension(
            'polynomial2d._poly2d',
            sources=['polynomial2d/_poly2d.c'],
            include_dirs=[np.get_include()],
        ),
    ],
    install_requires=open('requirements.txt').read().splitlines(),
)
