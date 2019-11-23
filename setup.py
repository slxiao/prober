"""prober -  Software bugs predictor based on machine learning techniques
"""
import io
import sys

import setuptools
from setuptools import setup

version = '0.1.1'

setup(
    include_package_data=True,
    name='prober',
    version=version,
    packages=['prober'],
    install_requires=[
    ],
    long_description=io.open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    entry_points={
        'console_scripts': [
            'prober = prober.prober:prober',
        ]
    },
    url='https://github.com/slxiao/prober',
    python_requires='>=2.6, !=3.0.*, !=3.1.*, !=3.2.*, <4',
    license='MIT',
    author='slxiao',
    author_email='shliangxiao@gmail.com',
    description='Software bugs predictor based on machine learning techniques',
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)