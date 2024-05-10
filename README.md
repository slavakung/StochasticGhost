# StochasticGhost
Algorithm for Solving Empirical Risk Minimization Problems with Constraints.
The module will be an installable dependency which is backend agnostic and can take an objective and any number of constraints.
The objective and constraints should be continuously differentiable functions.
The package is an implementation of a general purpose constrained optimization algorithm originally presented in https://arxiv.org/abs/2307.02943. This could be used for variety of problems for bias reduction, simulation of various engineering, mechanics and physics problems and so on. 
Unit tests are performed over MNIST and CIFAR-10 datasets which are available as python scripts in the test folder.
The ojective can be the loss of an ML model such as Neural Net and constraints can be some empirical risk measure.

## Setup Instructions

To install and use this package, follow these steps:

# 1. Clone this repository to your local machine:

   $git clone project_url
   $cd your_project

# 2. Create a virtual environment with python 3.10
  ### using python venv
   
   $python3.10 -m venv myenv

   Activate the virtual environment
   
   $source myenv/bin/activate

  ### using conda environments

   $conda create -n myenv python=3.10

   Activate the conda environment
   
   $source activate myenv

# 3. To build manually (installs all dependencies)

   $cd StochasticGhost
   
   $pip install wheel
   
   $python setup.py bdist_wheel
   
   $pip install dist/StochasticGhost-0.1.0-py3-none-any.whl

# To run the tests:
   ### Help
   $python income.py --help
   ### Run with a specific backend/model
   $python income.py --model "pytorch_connect"
