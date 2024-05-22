import os
from setuptools import setup, find_packages
from setuptools.extension import Extension

setup(
    name='StochasticGhost',
    version='0.1.0',
    description='Empirical Risk minimization with constraints',
    author='Vyacheslav Kungrutsev',
    author_email='kunguvya@fel.cvut.cz',
    packages=find_packages(),
    python_requires='==3.10.*',
    install_requires=[
        'setuptools',
        'numpy',
        'pot==0.9.3',
        'argparse',
        'scipy==1.11.4',
        'qpsolvers==4.2.0',
        'scikit-learn==1.3.0',
        'matplotlib',
        'autoray==0.6.7',
    ],
)
