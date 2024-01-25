from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import StochasticGhost
import numpy as np
from matplotlib import pyplot as plt
import sys
from .. import StochasticGhost
# Load CIFAR-10 data
from torchvision import datasets, transforms


# Load CIFAR-10 data
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# For simplicity, let's use only a subset of the CIFAR-10 data (e.g., the first 1000 samples)
#num_samples = 1000

flattened_arrays = []

# Iterate over each tuple in trainset and append the flattened array to the list
for data in trainset:
    flattened_array = np.reshape(data[0], -1)
    flattened_arrays.append(flattened_array)

xd = np.array(flattened_arrays)
yd = ([(data[1]) for data in trainset])

print(yd[0:10])

# Perform one-hot encoding for the labels
#yd_onehot = torch.eye(10)[yd]
yd = np.array(yd)

# Convert xd to numpy array for consistency with the original code
#xd = xd.numpy()

def paramvals(maxiter, beta, rho, lamb, hess, tau, mbsz, numcon, geomp, stepdecay, gammazero, zeta, N, n, lossbound):
    params = {
        'maxiter': maxiter,  # number of iterations performed
        'beta': beta,  # trust region size
        'rho': rho,  # trust region for feasibility subproblem
        'lamb': lamb,  # weight on the subfeasibility relaxation
        'hess': hess,  # method of computing the Hessian of the QP, options include 'diag' 'lbfgs' 'fisher' 'adamdiag' 'adagraddiag'
        'tau': tau,  # parameter for the hessian
        'mbsz': mbsz,  # the standard minibatch size, used for evaluating the progress of the objective and constraint
        'numcon': numcon,  # number of constraint functions
        'geomp': geomp,  # parameter for the geometric random variable defining the number of subproblem samples
        # strategy for step decrease, options include 'dimin' 'stepwise' 'slowdimin' 'constant'
        'stepdecay': stepdecay,
        'gammazero': gammazero,  # initial stepsize
        'zeta': zeta,  # parameter associated with the stepsize iteration
        'N': N,  # Train/val sample size
        'n': n,  # Total number of parameters
        'lossbound': lossbound  # Constraint bound
    }
    return params

class CustomNetwork(nn.Module):

    # For now the input data is passed as init parameters
    def __init__(self, layer_sizes, itrain, otrain_fives, otrain_ones):
        super(CustomNetwork, self).__init__()

        # Create a list of linear layers based on layer_sizes
        self.itrain = itrain
        self.otrain_fives = otrain_fives
        self.otrain_ones = otrain_ones
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

# Implementation of forward propagation function
    def forward(self, x):
        # self.input = x
        for layer in self.layers[:-1]:
            x = torch.sigmoid(layer(x))
        x = torch.sigmoid(self.layers[-1](x))
        return x

    def compute_loss(self, Y, Y_hat):
        L_sum = 0.5*torch.sum(torch.square(Y - Y_hat))

        m = Y.shape[0]
        # print("Y shape is: ", m)
        L = (1./m) * L_sum

        return L

# objective function definition
    def obj_fun(self, params, minibatch):
        # with torch.no_grad():
        x = self.itrain
        y_ones = self.otrain_ones
        model_parameters = list(self.parameters())
        samples = np.random.choice(len(y_ones), minibatch, replace=False)
        model_parameters[0].data = torch.Tensor(params[0])
        model_parameters[1].data = torch.Tensor(params[1])
        model_parameters[2].data = torch.Tensor(params[2])
        model_parameters[3].data = torch.Tensor(params[3])
        obj_fwd = self.forward(x[samples, :]).flatten()
        print(x[samples, :].shape)
        print(obj_fwd.shape)
        print(y_ones[samples].shape)
        # print("Predicted obj:", obj_fwd)
        # print("Actual obj", y_ones[samples])
        fval = self.compute_loss(obj_fwd, y_ones[samples])
        print("Objective loss for ones:", fval.item())
        return fval.item()

# Function to calculate the gradients of learnable parameters w.r.t. objective function
    def obj_grad(self, params, minibatch):
        # f_grad = {}
        fgrad = []
        x = self.itrain
        y_ones = self.otrain_ones
        model_parameters = list(self.parameters())
        samples = np.random.choice(len(y_ones), minibatch, replace=False)
        # with torch.no_grad():
        model_parameters[0].data = torch.Tensor(params[0])
        model_parameters[1].data = torch.Tensor(params[1])
        model_parameters[2].data = torch.Tensor(params[2])
        model_parameters[3].data = torch.Tensor(params[3])
        obj_fwd = self.forward(x[samples, :]).flatten()
        obj_loss = self.compute_loss(obj_fwd, y_ones[samples])

        obj_loss.backward()

        max_norm = 0.5
        clip_grad_norm_(self.parameters(), max_norm)
        for param in self.parameters():
            if param.grad is not None:
                # Clone to avoid modifying the original tensor
                fgrad.append(param.grad.data.clone().view(-1))

        # Manually set gradients to zero
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.zero_()

        fgrad.insert(4, torch.Tensor(params[4]).view(-1))
        fgrad.insert(5, torch.Tensor(params[5]).view(-1))

        fgrad = torch.cat(fgrad, dim=0)
        return fgrad

# constraint function definition
    def conf(self, params, minibatch):
        conf_val = None
        x = self.itrain
        y_fives = self.otrain_fives
        model_parameters = list(self.parameters())
        samples = np.random.choice(len(y_fives), minibatch, replace=False)
        model_parameters[0].data = torch.Tensor(params[0])
        model_parameters[1].data = torch.Tensor(params[1])
        model_parameters[2].data = torch.Tensor(params[4])
        model_parameters[3].data = torch.Tensor(params[5])
        conf_fwd = self.forward(x[samples, :]).flatten()
        # print("Predicted conf:",conf_fwd)
        # print("Actual conf",y_fives[samples])
        conf_val = self.compute_loss(conf_fwd, y_fives[samples])
        print("Constraint loss for fives:", conf_val.item())
        return conf_val.item()


# Function to calculate the gradients of learnable parameters w.r.t. constraint function
    def conJ(self, params, minibatch):
        x = self.itrain
        y_fives = self.otrain_fives
        cgrad = []
        model_parameters = list(self.parameters())
        samples = np.random.choice(len(y_fives), minibatch, replace=False)
        model_parameters[0].data = torch.Tensor(params[0])
        model_parameters[1].data = torch.Tensor(params[1])
        model_parameters[2].data = torch.Tensor(params[4])
        model_parameters[3].data = torch.Tensor(params[5])
        # with torch.no_grad():
        conf_fwd = self.forward(x[samples, :])
        conf_loss = self.compute_loss(conf_fwd, y_fives[samples])

        conf_loss.backward()
        max_norm = 0.5
        clip_grad_norm_(self.parameters(), max_norm)
        for param in self.parameters():
            if param.grad is not None:
                # Clone to avoid modifying the original tensor
                cgrad.append(param.grad.data.clone().view(-1))

        # Manually set gradients to zero without using optimizer.zero_grad()
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.zero_()

        cgrad.insert(2, torch.Tensor(params[2]).view(-1))
        cgrad.insert(3, torch.Tensor(params[3]).view(-1))

        cgrad = torch.cat(cgrad, dim=0)

        return cgrad
    
# Update the input size and output size of the neural network
ip_size = 200  # CIFAR-10 grayscale images are 32x32 pixels
hid_size1 = 32
op_size = 1  # There are 10 classes in CIFAR-10
layer_sizes = [ip_size, hid_size1, op_size]

# Selectively setting all output labels to zero except for class 1 and 5 in 2 different output tensors
#yd = np.array(yd)
#yd = yd.astype(int)
yfives = np.zeros((yd.shape[0],))
yones = np.zeros((yd.shape[0],))
yfives[np.where(yd == 1)] = 1
yones[np.where(yd == 5)] = 1

net = CustomNetwork(layer_sizes, xd[:, :ip_size], yfives, yones)


X_train = torch.tensor(xd)
Y_train_ones = torch.tensor(yones)
Y_train_fives = torch.tensor(yfives)
# cons_bound = 10.
trials = 21
maxiter = 100
ftrial = np.zeros((maxiter, trials))
ctrial = np.zeros((maxiter, trials))
initsaved = []
for trial in range(trials):
    print(">>>>>>>>>>>>>>>>>>>>>>TRIAL", trial+1)
# X_train = torch.tensor()
    net = CustomNetwork(layer_sizes, X_train[:, :ip_size], Y_train_fives, Y_train_ones)
    nn_parameters = list(net.parameters())
    initw = [param.data for param in nn_parameters]
    initw = initw + [nn_parameters[2].data, nn_parameters[3].data]
    num_param = sum(p.numel() for p in net.parameters()) + \
        nn_parameters[2].numel() + nn_parameters[3].numel()
    params = paramvals(maxiter=maxiter, beta=10, rho=0.8, lamb=0.5, hess='diag', tau=1., mbsz=100,
                       numcon=1, geomp=0.5, stepdecay='dimin', gammazero=0.1, zeta=0.1, N=X_train.shape[0], n=num_param, lossbound=[0.13])
    w, iterfs, itercs = StochasticGhost.StochasticGhost(
        net.obj_fun, net.obj_grad, [net.conf], [net.conJ], initw, params)
    ftrial[:, trial] = iterfs
    ctrial[:, trial] = itercs[:,0]


# Plotting the interquartile loss and constraint over number of iterations 
for i in range(0, maxiter):
    ftrial[i, :] = np.sort(ftrial[i, :])
    ctrial[i, :] = np.sort(ctrial[i, :])

plt.plot(range(0, maxiter), ctrial[:, 10], 'k-')
plt.fill_between(range(0, maxiter), ctrial[:, 5], ctrial[:, 15])
# plt.plot(range(0, maxiter+1), convals[:, 0], 'k-')
# plt.fill_between(range(0, maxiter+1), convals[:, 0], convals[:, 1])
plt.axhline(y=0.13, color='black', linestyle='--')
plt.title("Loss for CIFAR object 5")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('Constraint_loss_2fig.pdf')
# plt.yscale('log')
plt.clf()

plt.plot(range(0, maxiter), ftrial[:, 10], 'k-')
plt.fill_between(range(0, maxiter), ftrial[:, 5], ftrial[:, 15])
# plt.plot(range(0, maxiter+1), convals[:, 0], 'k-')
# plt.fill_between(range(0, maxiter+1), convals[:, 0], convals[:, 1])
plt.title("Loss for CIFAR object 1")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('Objective_loss_2fig.pdf')
# plt.yscale('log')
plt.clf()

