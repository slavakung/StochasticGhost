# StochasticGhost
Algorithm for Solving Empirical Risk Minimization Problems with Constraints.
The module will be an installable dependency which is backend agnostic and can take an objective and any number of constraints.
The objective and constraints should be continuously differentiable functions.
The package has a series of notebooks which show the algorithm's performance on benchmark datasets like: Propublica COMPAS, Adult Income.
Unit tests are also performed over MNIST and CIFAR-10 datasets which are available as python scripts in the test folder.
The ojective can be the loss of a ML model such as Neural Net and constraints can be some empirical risk measure.
