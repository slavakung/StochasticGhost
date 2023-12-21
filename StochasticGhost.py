from ot.utils import list_to_array
from ot.backend import get_backend
import warnings
import argparse
import numpy as np
import time
from scipy.optimize import linprog
from qpsolvers import solve_qp
from ot.utils import unif, dist, list_to_array
import autoray as ar
# from .backend import get_backend


def makeparms(maxiter=1, beta=10, rho=0.8, lamb=0.5, hess='diag', tau=1., mbsz=1, numcon=1, geomp=0.7, stepdecay='dimin', gammazero=0.1, zeta=0.1):
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
    }
    return params


def computekappa( cval, cgrad, rho, lamb, mc, n):
    obj = np.concatenate(([1.], np.zeros((n,))))
    Aubt = np.concatenate((([-1.]), cgrad))
    # if there are multiple constraints? Aubt.reshape(mc,n+1) ??
    Aubt = Aubt.reshape(mc, n+1)
    res = linprog(c=obj, A_ub=Aubt, b_ub=[-cval], bounds=(-rho, rho))
    return ((1-lamb)*max(0, cval)+lamb*max(0, res.fun))


def solvesubp(fgrad, cval, cgrad, kap, beta, tau, hesstype, mc, n):
    if hesstype == 'diag':
       #P = tau*nx.eye(n)
       P = tau*np.identity(n)
    # reshaping cgrad to (1,n), shouldn't it be more generalized? i.e. (mc, n) and if we are passing the cgrad as Jeval, it will automatically be that shape
    return solve_qp(P, fgrad.reshape((n,)), cgrad.reshape((mc, n)), list_to_array([(kap-cval)]), np.zeros((0, n)), np.zeros((0,)), -beta*np.ones((n,)), beta*np.ones((n,)), solver='osqp')


# initw : Initial parameters of the Network (Weights and Biases)

# Should pass a network Object ?
# For obj_function evaluation in each iteration, we need to do a forward pass in the NN
# Back and forth function calls
def StochasticGhost(obj_fun, obj_grad, con_funs, con_grads, initw, params):
    # N = params["N"] # Total train/val samples
    # print(params["N"])
    N = params["N"]
    n = params["n"]  # Total network parameters
    maxiter = params["maxiter"]
    beta = params["beta"]
    rho = params["rho"]
    lamb = params["lamb"]
    tau = params["tau"]
    hess = params["hess"]
    mbsz = params["mbsz"]
    mc = params["numcon"]
    geomp = params["geomp"]
    stepdec = params["stepdecay"]
    gamma0 = params["gammazero"]
    zeta = params["zeta"]
    gamma = gamma0

    #initw = list_to_array(initw)
    #nx = get_backend(initw)
    #w = nx.copy(initw)
    #print("The params:", w)
    w = initw
    for i in range(len(w)):
        w[i] = ar.to_numpy(w[i])
        # print(w[i].size)
    # feval = net_obj.forward_utility(w,mbsz)
    # forward_utility defined in the Model class then calls forward with the updated weights
    # After every iteration, have to go back and update the weights of network manually in the Model Class?
    # network.named_parameter.weight0 = w[0] ..... so on

    feval = obj_fun(w, mbsz)  # returns a tensor(Should return a generic type)
    # fgrad = nx.zeros((n,))
    ceval = np.zeros((mc,))
    Jeval = np.zeros((mc, n))

    # Getting all the constraints
    iterfs = np.zeros((maxiter,))
    iterfs[0] = feval
    for i in range(mc):
       conf = con_funs[i]
       print("LOOKIE LOOKIE::::::::: ", conf(w, mbsz))
       ceval[i] = np.max(conf(w, mbsz), 0)
    itercs = np.zeros((maxiter,))
    itercs[0] = np.max(ceval)

    for iteration in range(0, maxiter):

        if stepdec == 'dimin':
           gamma = gamma0/(iteration+1)**zeta
        if stepdec == 'constant':
           gamma = gamma0
        if stepdec == 'slowdimin':
           gamma = gamma*(1-zeta*gamma)
        if stepdec == 'stepwise':
           gamma = gamma0 / (10**(int(iteration*zeta)))

        Nsamp = np.random.geometric(p=geomp)
        while (2**(Nsamp+1)) > N:
          Nsamp = np.random.geometric(p=geomp)

#        mbatch1 = np.random.choice(N, 1, replace=False)
#        mbatch2 = np.random.choice(N, 2**Nsamp, replace=False)
#        mbatch3 = np.random.choice(N, 2**Nsamp, replace=False)
#        mbatch4 = np.random.choice(N, 2**(Nsamp+1), replace=False)
        mbatches = [1, 2**Nsamp, 2**Nsamp, 2**(Nsamp+1)]
        dsols = np.zeros((4, n))

        for j in range(4):
          feval = obj_fun(w, mbatches[j])
          fgrad = ar.to_numpy(obj_grad(w, mbatches[j]))
          print(type(fgrad))
          # cval = []
          # cgrad = []
          for i in range(mc):
            # con_funs[i](conf) and con_grads[i](conJ) ith constraint and constraint grad
            conf = con_funs[i]
            conJ = con_grads[i]
            # ceval and Jeval are evaluations of ith constraint and constraint grads for the parameter values
            # nx.max(conf(w,mbatches[j]),0) to ensure the problem is always in the feasible region
            ceval[i] = np.max(conf(w, mbatches[j]), 0)
            Jeval[i, :] = ar.to_numpy(conJ(w, mbatches[j]))
            #cons_grad = [ar.to_numpy(element) for element in cons_grad]
            #Jeval[i, :] = cons_grad
            print(type(Jeval[i, :]))
            # cval = nx.concatenate((cval, ceval[i]))
            # cgrad = nx.concatenate((cgrad, Jeval[i, :]))

          # expects cgrad as a 1-D array, but it will be (mc, n) shape array
          kap = computekappa(ceval[0], Jeval[0], rho, lamb, mc, n)
          print(type(kap), type(fgrad),
                type(ceval[0]), type(Jeval[0]))
          print(fgrad)
          print(Jeval[0])
          dsol = solvesubp(fgrad, ceval[0], Jeval[0], kap, beta, tau, hess, mc, n)
          dsols[j, :] = dsol

        dsol = dsols[0, :] + (dsols[3, :]-0.5*dsols[1, :] -
                              0.5*dsols[2, :])/(geomp*((1-geomp)**Nsamp))
        #print("Hey, I reached here and calculated the value of d :)")
        print(type(dsol))
        # print(dsol.shape)
        # print(w.shape)

        # w = w + gamma*dsol
        start = 0
        for i in range(len(w)):
           #print(w[i].size)
           end = start + np.size(w[i])
           w[i] = w[i] + gamma*np.reshape(dsol[start:end], np.shape(w[i]))
           start = end
        
        #print("Hey ", iteration+1, " iteration completed")
        feval = obj_fun(w, mbsz)
        iterfs[iteration] = feval
        for i in range(mc):
          conf = con_funs[i]
          ceval[i] = np.max(conf(w, mbsz), 0)
        itercs[iteration] = np.max(ceval)

    return w, iterfs, itercs
