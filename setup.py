import os
from setuptools import setup, find_packages
from setuptools.extension import Extension

setup(
    name='StochasticGhost',
    version='0.1.0',
    description='Empirical Risk minimization with constraints',
    author='Harsh Choudhary',
    author_email='choudharyharsh122@gmail.com',
    packages=find_packages(),
    python_requires='==3.10.13',
    install_requires=[
        'setuptools',
        'numpy',
        'pot==0.9.3',
        'warnings',
        'argparse',
        'time',
        'scipy=1.11.4',
        'qpsolvers=4.2.0',
        'scikit-learn=1.3.0',
        'matplotlib',
        'autoray=0.6.7',
    ],
)
