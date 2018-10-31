#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ 'pandas>=0.23.4', 'scipy>=1.1.0', 'matplotlib>=2.2.3', 'seaborn>=0.9.0','scikit-learn>=0.19.2', 'MulticoreTSNE>=0.0.1.1' ]
#requirements = [ 'pandas' ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Luis Aparicio",
    author_email='la2666@cumc.columbia.edu',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="A Library for Denoising Single-Cell Data with Random Matrix Theory",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='randomly',
    name='randomly',
    packages=find_packages(include=['randomly']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/RabadanLab/randomly',
    version='0.1.5',
    zip_safe=False,
)
