import numpy as np

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy import stats

from collections import deque

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Patch

import ipywidgets as wgs
from ipywidgets import interact, interactive
from IPython.display import display


from IPython.utils.path import ensure_dir_exists
fdir = 'figs/'
ensure_dir_exists(fdir)
resdir = 'res/'
ensure_dir_exists(resdir)


class Model:
    def __init__(self, name, params, freeparams):
        self.name = name
        self.params = params
        self.freeparams = freeparams

        
class ModelSSA(Model):
    formulation = 'SSA'
    
    def __init__(self, name, params, freeparams,
                 update_states=None, update_propensities=None,
                 check_break=lambda t, S, I, R, D: (I.sum() == 0)):
        super().__init__(name, params, freeparams)
        self.update_states = update_states
        self.update_propensities = update_propensities
        self.check_break = check_break
        
    def simulate(self, fparams):
        return Gillespie(updateStates=self.update_states, updatePropensities=self.update_propensities,
                         checkBreak=self.check_break,
                         **self.params, **fparams)
    
    def plot(self, N, T, t, S, I, R, D, ptitle=None, figsize=(14,8), plegend=False):
        return plotFilled(N, T, t, S, I, R, D, ptitle=ptitle, figsize=figsize, plegend=plegend)

    
class ModelODE(Model):
    formulation = 'ODE'
    
    def __init__(self, name, params, freeparams,
                 deriv=None):
        super().__init__(name, params, freeparams)
        self.deriv = deriv
        
    def simulate(self, fparams):
        return solveODE(deriv=self.deriv, **self.params, **fparams)
    
    def plot(self, N, T, t, S, I, R, D, ptitle=None, figsize=(14,8), plegend=False):
        return plotFilledODE(N, T, t, S, I, R, D, ptitle=ptitle, figsize=figsize, plegend=plegend)    
    
    
    
##-----------------------------

def Gillespie(T=100., N=100, frac=np.array([1]),
              Sinit=None, Iinit=np.array([1]), Rinit=None, Dinit=None,
              seed=None,
              updateStates=lambda rExc, S, I, R, D: (S, I, R, D),
              updatePropensities=lambda S, I, R, D, N, *args, **kwargs: np.ones(1),
              checkBreak=lambda t, S, I, R, D: (I.sum() == 0),
              **kwargs
             ):
    # Inital values
    t = 0.
    I = Iinit.copy()

    if Sinit is None:
        S = np.rint(N*frac - I).astype(int)
    else:
        S = Sinit.copy()
    if Rinit is None:
        R = np.zeros_like(I)
    else:
        R = Rinit.copy()
    if Dinit is None:
        D = np.zeros_like(I)
    else:
        D = Dinit.copy()  

    # Check for parameters
    for param in kwargs.values():
        if param < 0.:
            return (np.array([0., T]), np.tile(np.nan * S,(2,)), np.tile(np.nan * I,(2,)), np.tile(np.nan * R,(2,)), np.tile(np.nan * D,(2,)))

    # Vectors for time series
    time = [t]
    nS   = [S.copy()]
    nI   = [I.copy()]
    nR   = [R.copy()]
    nD   = [D.copy()]



    # Random Generator
    rg = np.random.RandomState(seed)

    # Main loop
    while t < T:
        if checkBreak(t, S, I, R, D): # no infectives any more means epidemic dies out
            break

        # First uniformily distributed ranmdom number for reaction time
        r1  = rg.uniform(0.0, 1.0)

        # Propensities calculation
        kappa = updatePropensities(S, I, R, D, N, **kwargs)
        Phi   = kappa.sum()
        #print(np.cumsum(kappa) / Phi)

        # waiting time tau
        tau = -np.log(r1) / Phi
        t   = t + tau

        # Second uniformily distributed random number for executed reaction
        r2 = rg.uniform(0.0, 1.0)

        # Rule determined to be executed
        bins = np.cumsum(kappa) / Phi
        rExc = np.searchsorted(bins, r2, side='right')

        # Update states
        S, I, R, D = updateStates(rExc, S, I, R, D)

        # Append new values
        time.append(t)
        nS.append(S.copy())
        nI.append(I.copy())
        nR.append(R.copy())
        nD.append(D.copy())

    time = np.array(time)
    nS   = np.array(nS)
    nI   = np.array(nI)
    nR   = np.array(nR)
    nD   = np.array(nD)
    return (time, nS, nI, nR, nD)     
    
def solveODE(T=100., N=100, frac=np.array([1]),
             Sinit=None, Iinit=np.array([1]), Rinit=None, Dinit=None, Uinit=None,
             deriv=lambda t, Y, ny, **kwargs: np.zeros(4*ny),
             **kwargs
            ):
    t = 0.
    
    I0 = Iinit/N
    if Sinit is None:
        S0 = frac - I0
    else:
        S0 = Sinit.copy()/N
    if Rinit is None:
        R0 = np.zeros_like(I0)
    else:
        R0 = Rinit.copy()/N
    if Dinit is None:
        D0 = np.zeros_like(I0)
    else:
        D0 = Dinit.copy()/N
    
    ny = [S0.size, I0.size, R0.size, D0.size] # number of subtypes
        
    Y0 = np.hstack((S0, I0, R0, D0))
    
    sol = solve_ivp(lambda t, Y: deriv(t, Y, ny, **kwargs), y0=Y0, t_span=(0, T), t_eval=np.arange(T))#, dense_output=True)
    t = sol.t
    Y = sol.y
    
    return (t, *map(np.squeeze, np.split(N*Y.T, np.cumsum(ny)[:-1], axis=1)))
    #return (t, *np.split(N*Y.T, np.cumsum(ny)[:-1], axis=1))
    
    
    
    
##----------

def plotFilled(N, T, time, nS, nI, nR, nD,
               tested=None,
               ptitle=None, plegend=True, pplot=True,
               figsize=(20,8),
               colors=['tab:orange', 'tab:green', 'tab:blue', 'black'],
               step='post'
              ):
    nt = 4

    typenames  = ["I", "S", "R", "D"]
    
    if tested is not None:
        nSt = nS[:,tested]
        nIt = nI[:,tested]
        nRt = nR[:,tested]
        nDt = nD[:,tested]
        nS = nS.sum(axis=1)
        nI = nI.sum(axis=1)
        nR = nR.sum(axis=1)
        nD = nD.sum(axis=1)
        nT  = np.column_stack((nIt, nSt, nRt, nDt))
    
    Ty = [nI, nS, nR, nD]
    
    nys = np.array([1 if len(t.shape) < 2 else t.shape[1] for t in Ty])
    nX  = np.c_[nI, nS, nR, nD]
    nXS = np.cumsum(nX, axis=1)
    nXS = np.hstack((np.zeros((nXS.shape[0], 1)),nXS)) # adding zero basis
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y = 0
    for ty in range(nt):
        ny = nys[ty]
        for sy in range(ny):
            y += 1
            ax.fill_between(time, nXS[:,y-1], nXS[:,y],
                            step=step,
                            color = colors[ty],
                            alpha = 0.8 - (ny-sy-1)*0.4/max(1,(ny-1)),
                            label = f'${typenames[ty]}_{sy+1}$' if ny > 1 else f'${typenames[ty]}$')
        if tested is not None:
            ax.fill_between(time, nXS[:, y-1], nXS[:, y-1]+nT[:,y-1], # from lower line upwards
            #ax.fill_between(time, nXS[:, ty*ny], nXS[:, ty*ny]-nT[:,ty*ny], # from upper line downwards
                            step=step,
                            edgecolor = 'white',#'black',#colors[ty],
                            #linewidth = 0,
                            facecolor = 'None',
                            alpha = 0.4,
                            hatch = '//')
            
    for i in range(nys.sum()):
        ax.text(min(T,time[-1]), 0.5*(nXS[-1,i]+nXS[-1,i+1]), "{:3.0f} % ".format(100*nX[-1,i]/N), ha='right',
                path_effects=[path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

        
    ax.set_xlim((0,T))
    ax.set_ylim((0,N))        
    
    if plegend:
        handles, labels = ax.get_legend_handles_labels()
        handles, labels = handles[::-1], labels[::-1]
        if tested is not None:
            handles.append(Patch(facecolor = 'gray',#'None',
                 edgecolor = 'white',#'black',
                 hatch = '//',
                 alpha = 0.4))
            labels.append(f'tested')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        legend = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), shadow=False, fontsize='x-large', frameon=False)
    if ptitle is not None:
        ax.set_title(ptitle)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Individuals')
    if pplot:
        plt.show(); 
    return fig

def plotFilledODE(N, T, time, nS, nI, nR, nD,
                  tested=None,
                  ptitle=None, plegend=True, pplot=True,
                  figsize=(20,8),
                  colors=['red', 'green', 'blue', 'gray'],
                  step=None):
    return plotFilled(N, T, time, nS, nI, nR, nD, tested, ptitle, plegend, pplot, figsize, colors, step)


##----

def vanillaABC(model, data, priors, tolerance, distance, statistics, M):
    freeparamsKeys = list(model.freeparams.keys())
    nfreeparams = len(freeparamsKeys) 
    
    samples = np.zeros((M,nfreeparams))
    distances = np.zeros(M)
    
    m, z = 0, 0
    while m < M:

        # sample parameters
        theta = np.atleast_1d(priors.rvs())

        ts, Ss, Is, Rs, Ds = model.simulate(dict(zip(freeparamsKeys, theta)))
        z += 1
        
        ds = distance(ts, Ss, Is, Rs, Ds)
        print(f'(acc.: {m:8} / total: {z:8})', end='\r')
        
        if ds <= tolerance:
            samples[m] = np.squeeze(theta)
            distances[m] = ds
            m += 1

    return np.squeeze(samples), distances, z

def ABC_MCMC(model, data, priors, tolerance, distance, statistics, proposal, paraminit, M, K, scale=0.1):
    freeparams = dict.fromkeys(model.freeparams.keys())
    nfreeparams = len(freeparams) 
    m, z = 0, 0
    samples = np.zeros((M, nfreeparams))
    distances = np.zeros(M)
    Hs = np.zeros(M)
    totalsamples = []
    
    thetaC = np.atleast_1d(paraminit)
    rC = 1.
    
    while m < M:
        print(f'({m:4} / {z:4})', end='\r')
        
        #sample from proposal q(.|thetaC)
        theta = np.atleast_1d(proposal.rvs(thetaC, scale))
        
        rk = 0.
        dk = 0.
        for k in range(K):
            ts, Ss, Is, Rs, Ds = model.simulate(dict(zip(freeparams.keys(), theta)))
            z += 1
            
            ds = distance(ts, Ss, Is, Rs, Ds)

            if ds < tolerance:
                rk += 1
                dk += ds 
        r = rk/K
        d = dk/K
        
        H = min(1., r/rC * proposal.pdf(thetaC, theta, scale)/proposal.pdf(theta, thetaC, scale))
                #priors.pdf(theta)/priors.pdf(thetaC) * 
        
        #print(H)
        if stats.uniform.rvs() <= H:
            thetaC = theta
            rC = r
            
            samples[m] = thetaC
            distances[m] = d
            m += 1
        totalsamples.append(thetaC)
    
    return np.squeeze(samples), np.squeeze(distances), z, np.squeeze(np.array(totalsamples))

def M_ABC_SMC(models, modelprior, data, priors, tolerances, distance, statistics, kernel, S, K, scale=0.1):
    G = len(tolerances)
    M = len(models)
    print(G)
    
    freeparamsKeys = []
    nfreeparams = np.zeros(M, dtype=int)
    for m in range(M):
        freeparamsKeys.append(list(models[m].freeparams.keys()))
        nfreeparams[m] = len(freeparamsKeys[m]) 
      
    samples = [np.full((G, S, nfreeparams[m]), np.nan) for m in range(M)]
    weights = np.full((M, G, S), np.nan) 
    msamples = np.full((G, S), np.nan) 
    
    g = 0
    zs = np.zeros(G, dtype=int)
    z = 0
    
    while g < G:
        s = 0
        while s < S:   
            if g == 0:
                m = modelprior.rvs()
            else:
                m = modelprior.rvs()
                #print('here',m,  np.all(np.isnan(weights[m, g-1, :]))) ###
                while np.all(np.isnan(weights[m, g-1, :])):
                    m = modelprior.rvs()
                    #print(m) ###
            ds = np.inf
            while np.isnan(ds) or  ds >= tolerances[g]:
                print(f'({g:4} / {s:4} / {z:4})', end='\r') ###
                # sample parameters for model m
                if g == 0:
                    thetaSP = np.atleast_1d(priors[m].rvs())
                else:
                    while True:
                        isample = np.random.choice(range(S), p=np.nan_to_num(weights[m, g-1, :]))
                        thetaS = samples[m][g-1,isample]
                        thetaSP = np.atleast_1d(kernel.rvs(thetaS, scale))
                        if np.all(priors[m].pdf(thetaSP) > 0.): break
                 
                ts, Ss, Is, Rs, Ds = models[m].simulate(dict(zip(freeparamsKeys[m], thetaSP)))
                zs[g] += 1
                z += 1
                
                ds = distance(ts, Ss, Is, Rs, Ds)

            samples[m][g, s, :] = thetaSP
            msamples[g, s] = m
            # Calculate weights
            if g == 0: # only for uniform priors!!
                weights[m, g, s] = 1.
            else:
                if np.any(weights[m, g-1, :] > 0.):
                    weights[m, g, s] = 1. / np.nansum(weights[m, g-1, :] * kernel.pdf(samples[m][g-1, :], thetaSP, scale))
            s += 1
        # Normalise weights
        for m in range(M):
            if np.any(weights[m, g, :] > 0.):
                weights[m, g, :] = weights[m, g, :] / np.nansum(weights[m, g, :])
        g += 1
    
    #return [np.squeeze(samps, axis=1) for samps in samples], weights, z
    return msamples, samples, weights, z