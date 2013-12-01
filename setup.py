#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup


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
    install_requires=open('requirements.txt').read().splitlines(),
)
