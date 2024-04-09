import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

# Define extension modules
extensions = [
    Extension('ghost.module1', ['your_library/module1.pyx']),
    Extension('ghost.module2', ['your_library/module2.cpp'],
              extra_compile_args=['-std=c++11']),
]

# Cythonize .pyx files
extensions = cythonize(extensions)

setup(
    name='StochasticGhost',
    version='0.1.0',
    description='Empirical Risk minimization with constraints',
    author='Harsh Choudhary',
    author_email='choudharyharsh122@gmail.com',
    packages=find_packages(),
    ext_modules=extensions,
    install_requires=[python_requires='>=3.10',
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
