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
    def __init__(self, layer_sizes, itrain, otrain):
        super(CustomNetwork, self).__init__()

        # Create a list of linear layers based on layer_sizes
        self.itrain = itrain
        self.otrain = otrain
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        # self.input = x
        for layer in self.layers[:-1]:
            x = torch.sigmoid(layer(x))
        x = torch.sigmoid(self.layers[-1](x))
        return x

    def compute_loss(self, Y_pred, Y_true):
        """
        compute loss function
        """
        # L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
        L_sum = 0.5*torch.sum(torch.square(Y_pred - Y_true))

        m = Y_true.shape[0]
        L = (1./m) * L_sum
        return L

    def obj_fun(self, params, minibatch):
        model_parameters = list(self.parameters())
        # with torch.no_grad():
        x = self.itrain
        y = self.otrain
        samples = np.random.choice(len(y), minibatch, replace=False)
        for i in range(len(params)):
            model_parameters[i].data = torch.Tensor(params[i])
        # self.compute_loss(fval, y[minibatch])
        obj_forward = self.forward(x[samples, :]).float().flatten()
        fval = self.compute_loss(obj_forward, y[samples])
        return fval.item()

    def obj_grad(self, params, minibatch):
        # f_grad = {}
        fgrad = []
        x = self.itrain
        y = self.otrain
        samples = np.random.choice(len(y), minibatch, replace=False)
        obj_forward = self.forward(x[samples, :]).float().flatten()
        obj_loss = self.compute_loss(obj_forward, y[samples])
        obj_loss.backward(retain_graph=True)

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

        # with torch.no_grad():
        # for name, param in self.named_parameters():
        #     fgrad.append(torch.autograd.grad(obj_loss, param, retain_graph=True)[0].view(-1))
        #     #fgrad.append(torch.autograd.grad(torch.sum(self.forward(x[samples, :])), param, retain_graph=True)[0].view(-1))
        fgrad = torch.cat(fgrad, dim=0)
        print(type(fgrad))
        return fgrad

    def conf(self, params, minibatch):
        conf_val = None
        # x_val = self.ival
        # y_val = self.oval
        # samples = np.random.choice(len(y_val), minibatch, replace=False)
        params = list(self.parameters())

# Extract the weights and biases
        W1 = params[0].reshape(-1)
        W2 = params[2].reshape(-1)
        # W3 = params[4].reshape(-1)
        B1 = params[1].reshape(-1)
        B2 = params[3].reshape(-1)
        # B3 = params[5].reshape(-1)

        # print("Weights1: ", W1)
        # print("Weights2: ", W2)
        # print("Bias1: ", B1)
        # print("Bias2: ", B2)

# Perform the operations in PyTorch
        conf_val = torch.dot(W1, W1) / 4. + torch.dot(W2, W2) / 4. + torch.dot(B1, B1) + torch.dot(B2, B2)
        """
        TODO: Compute
        
        """
        print("Constraint value: ", conf_val.item())
        return conf_val.item()

    def conJ(self, params, minibatch):
        # x_val = self.ival
        # y_val = self.oval
        # samples = np.random.choice(len(y_val), minibatch, replace=False)
        # with torch.no_grad():
        # for name, param in self.named_parameters():
        # print(name)
        params = list(self.parameters())
        cgrad = [(params[0]/2.).reshape(-1), 2*params[1].reshape(-1), (params[2] / 2.).reshape(-1), 2*params[3].reshape(-1)]

        cgrad = torch.cat(cgrad, dim=0)
        # print(cgrad)
        # print(cgrad.shape)
        """"
        TODO: Compute

        """
        # print("Constraint gradient is: ", cgrad)
        return cgrad

ip_size = 200
hid_size1 = 64
op_size = 1
layer_sizes = [ip_size, hid_size1, op_size] 

X_train = torch.tensor(xd)
Y_train = torch.tensor(yd)
print("Train data size:", len(Y_train))
# cons_bound = 10.
trials = 21
maxiter = 100
ftrial = np.zeros((maxiter, trials))
ctrial = np.zeros((maxiter, trials))
for trial in range(trials):  # X_train = torch.tensor()
    print(">>>>>>>>>>>>>>>>>TRIAL:", trial+1)
    net = CustomNetwork(
        layer_sizes, X_train[:, :ip_size], Y_train)
    nn_parameters = list(net.parameters())
    initw = [param.data for param in nn_parameters]
    num_param = sum(p.numel() for p in net.parameters())
    # net.train_and_update(xd, yd)
    # output_t = net.forward(X_train[:, :70])
    # output_v = net.forward(X_val[:, :70])
    # obj = net.compute_loss(Y_train[:70], output_t)
    # constraint = output_v-cons_bound
    params = paramvals(maxiter=maxiter, beta=10, rho=1., lamb=0.5, hess='diag', tau=1., mbsz=100,
                       numcon=1, geomp=0.5, stepdecay='dimin', gammazero=0.1, zeta=0.1, N=X_train.shape[0], n=num_param, lossbound=[5])
    w, iterfs, itercs = StochasticGhost.StochasticGhost(
        net.obj_fun, net.obj_grad, [net.conf], [net.conJ], initw, params)
    ftrial[:, trial] = iterfs
    ctrial[:, trial] = itercs[:,0]


for i in range(0, maxiter):
    ftrial[i, :] = np.sort(ftrial[i, :])
    ctrial[i, :] = np.sort(ctrial[i, :])

plt.plot(range(0, maxiter), ctrial[:, 10], 'k-')
plt.fill_between(range(0, maxiter), ctrial[:, 5], ctrial[:, 15])
plt.axhline(y=5, color='black', linestyle='--')
# plt.plot(range(0, maxiter+1), convals[:, 0], 'k-')
# plt.fill_between(range(0, maxiter+1), convals[:, 0], convals[:, 1])
plt.title("Constraint Value and Threshold")
plt.xlabel('Iteration')
plt.ylabel('Constraint Value')
plt.savefig('Constraint_Loss_elliptic.pdf')
plt.clf()

plt.plot(range(0, maxiter), ftrial[:, 10], 'k-')
plt.fill_between(range(0, maxiter), ftrial[:, 5], ftrial[:, 15])
# plt.plot(range(0, maxiter+1), convals[:, 0], 'k-')
# plt.fill_between(range(0, maxiter+1), convals[:, 0], convals[:, 1])
plt.title("Loss for CIFAR dataset")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('Objective_loss_elliptic.pdf')
plt.clf()
