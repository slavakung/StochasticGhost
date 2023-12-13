from ot.utils import list_to_array
from ot.backend import get_backend
import warnings
import argparse
import numpy as np
import time
from scipy.optimize import linprog
from qpsolvers import solve_qp
from ot.utils import unif, dist, list_to_array
from .backend import get_backend

def makeparms(maxiter = 100, beta = 10, rho = 0.8, lamb = 0.5, hess = 'diag', tau = 1., mbsz = 1, numcon = 1, geomp=0.7, stepdec='dimin',gammazero=0.1,zeta=0.1):
    params = {
       'maxiter' : maxiter,  #number of iterations performed
       'beta'    : beta, # trust region size
       'rho'     : rho, # trust region for feasibility subproblem
       'lamb'    : lamb, # weight on the subfeasibility relaxation
       'hess'    : hess, # method of computing the Hessian of the QP, options include 'diag' 'lbfgs' 'fisher' 'adamdiag' 'adagraddiag'
       'tau'     : tau, # parameter for the hessian
       'mbsz'    : mbsz,  # the standard minibatch size, used for evaluating the progress of the objective and constraint
       'numcon'  : numcon, # number of constraint functions
       'geomp'   : geomp, # parameter for the geometric random variable defining the number of subproblem samples
       'stepdec' : stepdecay, # strategy for step decrease, options include 'dimin' 'stepwise' 'slowdimin' 'constant'
       'gammmazero' : gammazero, # initial stepsize
       'zeta'    : zeta, # parameter associated with the stepsize iteration
    }
    return params


def computekappa(nx, cval, cgrad, rho, lamb, n):
    obj = nx.concatenate(([1.],nx.zeros((n,))))
    Aubt = nx.concatenate((([-1.]),cgrad))
    Aubt = Aubt.reshape(1,n+1)
    res = linprog(c=obj, A_ub=Aubt,b_ub=[-cval], bounds=(-rho,rho))    
    return ((1-lamb)*max(0,cval)+lamb*max(0,res.fun)) 
    

def solvesubp(nx, fgrad, cval, cgrad, kap, beta, tau, hesstype, n):
    if hesstype == 'diag':
       P = tau*nx.eye(n)

    return solve_qp(P, fgrad.reshape((n,)), cgrad.reshape((1,n)), nx.array([(kap-cval)]), nx.zeros((0,n)), nx.zeros((0,)), -beta*nx.ones((n,)),beta*nx.ones((n,))) 
    

def StochasticGhost(obj_fun, obj_grad, con_funs, con_grads, initw, params):
    N = params.N
    n = params.n
    maxiter = params.maxiter
    beta = params.beta
    rho = params.rho
    lamb = params.lamb
    tau = params.tau
    hess = params.hess
    mbsz = params.mbsz 
    mc = params.numcon
    geomp = params.geomp
    stepdec = params.stepdecay
    gamma0 = params.gammazero
    zeta = params.zeto
    gamma = gamma0
    
    nx = backend.get_backend(initw)
    w = nx.copy(initw)
    feval = obj_fun(w,mbsz)
    ceval = nx.zeros((mc,))
    Jeval = nx.zeros((mc,n))
    
    iterfs = nx.zeros((maxiter,))
    iterfs[0] = feval    
    for i in range(mc):
       conf = con_funs[i]
       ceval[i] = nx.max(conf(w,mbsz),0)
    itercs = nx.zeros((maxiter,))
    itercs[0] = nx.max(ceval)
 
    for iteration in range(1,maxiter):
    
    
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
        mbatches = [1, 2**Nsamp,2**Nsamp,2**(Nsamp+1)]
        dsols = nx.zeros((4,n))

        for j in range(4):
          feval = obj_fun(w,mbatches[j])
          geval = obj_grad(w,mbatches[j])
          for i in range(mc):
            conf = con_funs[i]
            conJ = con_grads[i]
            ceval[i] = nx.max(conf(w,mbatches[j]),0)
            Jeval[i,:] = conJ(w,mbatches[j])
 
          kap = computekappa(nx, cval, cgrad, rho, lamb, n)
          dsol = solvesubp(nx, fgrad, cval, cgrad, kap, beta, tau, hesstype, n)
          dsols[j,:] = dsol

        dsol = dsols[0,:] + (dsols[3,:]-0.5*dsols[1,:]-0.5*dsols[2,:])/(geomp*((1-geomp)**Nsamp))      
        
        w = w + gamma*dsol
   
        feval = obj_fun(w,mbsz)
        iterfs[iteration] = feval    
        for i in range(mc):
          conf = con_funs[i]
          ceval[i] = nx.max(conf(w,mbsz),0)
        itercs[iteration] = nx.max(ceval)

    return w, iterfs, itercs


