import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

setup(
    name='StochasticGhost',
    version='0.1.0',
    description='Empirical Risk minimization with constraints',
    author='Harsh Choudhary',
    author_email='choudharyharsh122@gmail.com',
    packages=find_packages(),
    ext_modules=extensions,
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'ot',
        'warnings',
        'argparse',
        'time',
        'scipy',
        'qpsolvers',
        'scikit-learn',
        'matplotlib',
        'autoray',
    ]
    ],
)
