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
        'lossbound': lossbound # constraint bound
    }
    return params


class CustomNetwork(nn.Module):

    # For now the input data is passed as init parameters
    def __init__(self, layer_sizes, itrain, otrain, ival, oval):
        super(CustomNetwork, self).__init__()

        # Create a list of linear layers based on layer_sizes
        self.itrain = itrain
        self.otrain = otrain
        self.ival = ival
        self.oval = oval
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.sigmoid((layer(x)))
        x = torch.sigmoid(self.layers[-1](x))
        return x

    def compute_loss(self, Y, Y_hat):
        L_sum = 0.5*torch.sum(torch.square(Y - Y_hat))

        m = Y.shape[0]
        # print("Y shape is: ", m)
        L = (1./m) * L_sum

        return L

    def bce_logits(self, outputs, targets):
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(outputs, targets)
        if torch.isnan(loss).any():
            for name, param in self.named_parameters():
                print(name)
                print(param.data)
        return loss

    def obj_fun(self, params, minibatch):
        model_parameters = list(self.parameters())
        x = self.itrain
        y = self.otrain
        samples = np.random.choice(len(y), minibatch, replace=False)
        for i in range(len(params)):
            model_parameters[i].data = torch.Tensor(params[i])
        obj_fwd = self.forward(x[samples, :]).flatten()
        fval = self.compute_loss(obj_fwd, y[samples])
        print("Training loss is: ", fval)
        return fval.item()

    def obj_grad(self, params, minibatch):
        fgrad = []
        x = self.itrain
        y = self.otrain
        samples = np.random.choice(len(y), minibatch, replace=False)
        obj_fwd = self.forward(x[samples, :]).flatten()
        obj_loss = self.compute_loss(obj_fwd, y[samples])

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

        fgrad = torch.cat(fgrad, dim=0)
        return fgrad

    def conf(self, params, minibatch):
        # print("Reached at function constraint")
        conf_val = None
        x_val = self.ival
        y_val = self.oval
        samples = np.random.choice(len(y_val), minibatch, replace=False)
        # conf_val = self.forward(x_val[minibatch, :])
        cons_fwd = self.forward(x_val[samples, :]).flatten()
        conf_val = self.compute_loss(cons_fwd, y_val[samples])
        print("Validation loss is: ", conf_val)
        return conf_val.item()

    def conJ(self, params, minibatch):
        # print("Reached at function constraint grad")
        x_val = self.ival
        y_val = self.oval
        cgrad = []
        samples = np.random.choice(len(y_val), minibatch, replace=False)
        # with torch.no_grad():
        cons_fwd = self.forward(x_val[samples, :]).flatten()
        cons_loss = self.compute_loss(cons_fwd, y_val[samples])
        cons_loss.backward()
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

        cgrad = torch.cat(cgrad, dim=0)

        return cgrad

y = np.zeros(len(yd))
y[np.where(yd == 2)] = 1
print(len(xd))
print(len(y))


ip_size = 200
hid_size1 = 64
hid_size2 = 32
op_size = 1
layer_sizes = [ip_size, hid_size1, hid_size2, op_size]
X_train = torch.tensor(xd)
Y_train = torch.tensor(y)
# cons_bound = 10.
trials = 21
maxiter = 100
ftrial = np.zeros((maxiter, trials))
ctrial = np.zeros((maxiter, trials))
initsaved = []
for trial in range(trials):
    print(">>>>>>>>>>>>>>>>>TRIAL no:", trial+1)
    X_train, X_val, Y_train, Y_val = train_test_split(
        torch.tensor(xd), torch.tensor(y), test_size=0.4, random_state=42)
    train_loss_list = []
    val_loss_list = []
    # X_train = torch.tensor()
    net = CustomNetwork(
        layer_sizes, X_train[:, :ip_size], Y_train, X_val[:, :ip_size], Y_val)
    # net.apply(net.init_weights)
    nn_parameters = list(net.parameters())
    # print(net)
    initw = [param.data for param in nn_parameters]
    # print(len(initw))
    num_param = sum(p.numel() for p in net.parameters())
    params = paramvals(maxiter=maxiter, beta=10., rho=1., lamb=0.5, hess='diag', tau=1., mbsz=10,
                       numcon=1, geomp=0.5, stepdecay='dimin', gammazero=0.1, zeta=0.1, N=X_val.shape[0], n=num_param, lossbound=[0.1])
    w, iterfs, itercs = StochasticGhost.StochasticGhost(
        net.obj_fun, net.obj_grad, [net.conf], [net.conJ], initw, params)
    ftrial[:, trial] = iterfs
    ctrial[:, trial] = itercs[:,0]


for i in range(0, maxiter):
    ftrial[i, :] = np.sort(ftrial[i, :])
    ctrial[i, :] = np.sort(ctrial[i, :])

plt.plot(range(0, maxiter), ctrial[:, 10], 'k-')
plt.fill_between(range(0, maxiter), ctrial[:, 5], ctrial[:, 15])
plt.axhline(y=0.01, color='black', linestyle='--')
# plt.plot(range(0, maxiter+1), convals[:, 0], 'k-')
# plt.fill_between(range(0, maxiter+1), convals[:, 0], convals[:, 1])
plt.title("Validation loss for CIFAR object 2")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('Constraint_trainvalid.pdf')
plt.clf()

plt.plot(range(0, maxiter), ftrial[:, 10], 'k-')
plt.fill_between(range(0, maxiter), ftrial[:, 5], ftrial[:, 15])
# plt.plot(range(0, maxiter+1), convals[:, 0], 'k-')
# plt.fill_between(range(0, maxiter+1), convals[:, 0], convals[:, 1])
plt.title("Training loss for CIFAR object 2")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('Objective_trainvalid.pdf')
plt.clf()
