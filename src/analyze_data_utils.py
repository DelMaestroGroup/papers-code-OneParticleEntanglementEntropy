from multiprocessing.sharedctypes import Value
import numpy as np
from LL_utils import γ_V, γeq_V, v_a_timeres
from load_data_utils import QuenchParticleEE
from typing import List, Dict, Any, Tuple
import scipy
from tqdm import tqdm
from math import log
from scipy.signal import find_peaks
from numba import njit


def fit_tail_linear(x: np.ndarray, y: np.ndarray, npoints: int, std_err: np.ndarray = None):
    """Fits the first npoints curve y(x)=a*x+b with a linear model. If error bars are given,
    return also the std-error in b."""
    # fit y(x) = ax + b
    idx = np.argsort(x)

    datax = x[idx][:npoints+1]
    datay = y[idx][:npoints+1]
    if std_err is not None:
        err = 1/std_err[idx][:npoints+1]
        use = [True if not(np.isnan(y) or np.isinf(y) or np.isnan(
            e) or np.isinf(e)) else False for y, e in zip(datay, err)]
        err = err[use]
    else:
        use = [True if not(np.isnan(y) or np.isinf(y))
               else False for y in datay]
    datax = datax[use]
    datay = datay[use]

    if std_err is not None:
        if np.sum(use) == 0:
            return np.nan, np.nan, np.nan
        try:
            w = [x if not(np.isnan(x) or np.isinf(x)) else 1 for x in err]
            (a, b), C = np.polyfit(datax, datay, 1, w=w, cov="unscaled")
            err_b = np.sqrt(C[1, 1])
            return a, b, err_b
        except np.linalg.LinAlgError as e:
            print("Warning: ", e, " Return 0.0.")
            return 0.0, 0.0, 0.0

    try:
        if np.sum(use) == 0:
            return np.nan, np.nan
        a, b = np.polyfit(datax, datay, 1)
        return a, b
    except np.linalg.LinAlgError as e:
        print("Warning: ", e, " Return 0.0.")
        return 0.0, 0.0


def interpolate_entropy_tdlimit(Sα: np.ndarray, nFermions_array: np.ndarray, iα: int, fit_npoints: int = 5, V_array: np.ndarray = None):
    """Performs a linear fit to the first fit_npoints of the Sα_iα(1/nFermions_array) relation
    to obtain the thermodynamic limit estimation for nFermions --> inf. This fit is performed for each
    interaction strength in Sα[iV,iα,inFermions] but only the given Renyi power iα. (here iα = i evaluates Sα[:,i-1,:]
    to ensure that iα is the Renyi power).
    """
    nV = np.shape(Sα)[0]
    Sα_inf = np.zeros(nV)

    if iα == 0.5:
        iα = 11

    nα = np.shape(Sα)[1]
    assert 1 <= iα <= nα

    for iV in range(nV):
        _, b = fit_tail_linear(1/nFermions_array, Sα[iV, iα-1, :], fit_npoints)
        Sα_inf[iV] = b

    return Sα_inf


def optimize_ϵ_toED(numerics:np.ndarray, theory_fnc:callable, theory_args_list:List[Tuple], start_val: float = 1.0, no_dynamic_start_val: bool = False, optim_kwag: Dict = None):
    """Trys to optimize the first parameter of theory_fnc to fit data in numerics for each value
    in numerics. For each value i, numerics[i], use the tuple theory_args_list[i] as *args of
    the theory_fnc."""
    if optim_kwag is None:
        optim_kwag = dict()

    init_start_val = start_val
 
    res = []
    for numVal, args in zip(tqdm(numerics),theory_args_list):
        try:
            opt_fnc = lambda x,*args :  theory_fnc(x[0],*args)-numVal 
            res.append(scipy.optimize.root(fun=opt_fnc, x0=[start_val], args=args, options=dict(maxiter=200,disp=False),**optim_kwag))
            if not no_dynamic_start_val:
                try:
                    start_val = res[-1].x[0] if 0.5 < res[-1].x[0]  < 2 else init_start_val
                except:
                    pass
        except:
            res.append(np.nan)

    return res

def estimate_tinf_limit_all(quench:QuenchParticleEE,N:int):
    nRenyi = np.shape(quench.S)[0]
    nV = np.size(quench.V)

    Sinf = np.zeros((nRenyi,1,nV))
    Serr = np.zeros((nRenyi,nV))
    for iV,V in enumerate(quench.V):
        for iR in range(nRenyi):
            try: 
                Sinf[iR,0,iV],Serr[iR,iV] = intinite_t_entropy_and_error_sum(quench.t, np.squeeze(quench("Sind",i=iR,V=iV)) ,N,V,quench.Δt)
                if np.isnan(Sinf[iR,0,iV]) or np.isnan(Serr[iR,iV]): 
                    raise ValueError("nan value obtained in t->inf estimation")
            except:
                print(f"Warning for V={V}, iR={iR} error estimation failed.  Use fallback method.")
                Sinf[iR,0,iV],Serr[iR,iV],_,_ = intinite_t_entropy_and_error(quench.tres(V),np.squeeze(quench("Sind",i=iR,V=iV)),N)
            
    quench.S = np.concatenate((quench.S,Sinf),axis=1)
    quench.t = np.concatenate([quench.t,[np.inf]])
    quench.Serr = Serr 

    return quench


def estimate_tinf_tdlimit_all(quench_N:List[QuenchParticleEE],N_array:List[int],fit_npoints:int=5):
    for quench in quench_N:
        if not np.isinf(quench.t[-1]):
            raise ValueError("Before estimating the thermodynamic limit, the infinite time limit needs to be performed.")
    Vs = quench_N[0].V
    nV = np.size(Vs)
    nR = np.shape(quench_N[0].S)[0] 

    Sinf = np.zeros((nR,nV))
    Serr = np.zeros((nR,nV))
    
    for iR in range(nR):
        for iV,V  in enumerate(Vs):
            # get N dependence of infinite time limit
            SinfN = np.array([quench("Sind",i=iR,t=[-1],V=iV)[0] for quench in quench_N])
            SerrN = np.array([quench("Serrind",i=iR,V=iV) for quench in quench_N])
            # fit t->inf
            #print(f"V={V}, i={iR}, SinfN={SinfN}")
            _,Sinf[iR,iV],Serr[iR,iV] = fit_tail_linear(1/N_array,SinfN,npoints=fit_npoints,std_err=SerrN)
    return Vs,Sinf,Serr



def intinite_t_entropy_and_error_sum(t,EE,N,V,Δt):
    Einf,Eerr1  = intinite_t_entropy_and_error_1(t,EE,N,V,Δt)

    tres = t*v_a_timeres(V)
    _, Eerr2, _, _ = intinite_t_entropy_and_error(tres,EE,N)

    return Einf, Eerr1+Eerr2

def intinite_t_entropy_and_error_1(t,EE,N,V,Δt): 
    Einf,_ = get_asymptotic_value(t,np.squeeze(EE),N,V,Δt)
    _,_,Eerr, = binning_error(np.squeeze(EE))

    return Einf, Eerr

def binning_error(data):
    '''Perform a binning analysis'''
    
    from scipy.signal import find_peaks
    
    # number of possible binning levels
    num_levels = np.int(np.log2(data.shape[0]/4))+1

    # compute the error at each bin level
    Δ = []
    num_bins = []
    binned_data = data
    
    for n in range(num_levels):
        Δₙ,binned_data = get_binned_error(binned_data)
        Δ.append(Δₙ)
        num_bins.append(2**n) 
        
    Δ = np.array(Δ)
    
    # find the maxima which corresponds to the error
    if Δ.ndim == 1:
        plateau = find_peaks(Δ)[0]
        if plateau.size > 0:
            binned_error = Δ[plateau[0]]
        else:
            binned_error = np.max(Δ)
            print('Binning Converge Error: no plateau found')
    else:
        num_est = Δ.shape[1]
        binned_error = np.zeros(num_est)
        
        for iest in range(num_est):
            plateau = find_peaks(Δ[:,iest])[0]
            if plateau.size > 0:
                binned_error[iest] = Δ[plateau[0],iest]
        else:
            binned_error[iest] = np.max(Δ[:,iest])
            print('Binning Converge Error: no plateau found')
            
    return np.array(num_bins),Δ,binned_error

def get_binned_error(data):
    '''Get the standard error in mc_data and return neighbor averaged data.'''
    N_bins = data.shape[0]
    Δ = np.std(data,axis=0)/np.sqrt(N_bins)
    
    start_bin = N_bins % 2
    binned_data = 0.5*(data[start_bin::2]+data[start_bin+1::2])
    
    return Δ,binned_data

def get_asymptotic_value(t,EE,N,Vf,Δt):
    
    # get the index where we start the average
    idx = np.argmin(np.abs(t*v_a_timeres(Vf)-0.5*N)) 
    
    # find out how many data points we have in total to average
    num_times = len(t)
    
    # break into M pieces
    M = 4
    width = int(num_times/M)
    norm = 1#2/N
    asymp = []
    for i in range(M):
        start = idx + width*i
        end = start + width
        if i == M-1:
            end = len(t)
        asymp.append(np.average(EE[start:end])*norm)
    
    asymp = np.array(asymp)
    
    return np.average(asymp),np.std(asymp)/np.sqrt(M)

def intinite_t_entropy_and_error(tres,EE,N):
    """
    Obtain the infinite time limit and error.
    Input:
        tres: t*v/a_0 rescaled time for LL comparison
        EE: entanglement entropy
        N: number of fermions, system size L=2N 
    Returns:
        Einf: infinite time estimate of the entropy
        err: estimated error using bins std(Mi)/sqrt(Ni) + |mean(EE)-mean(Mi)|
        Mi: bin means of entropy
        x: considered rescaled time range
    """
    start_idx = np.argmin(np.abs(tres-2*N))
    x = tres[start_idx:]
    y = np.squeeze(EE)[start_idx:]

    # obtain infinite limit from mean
    Einf = np.mean(y)
    # obtain error by computing mean over every period of size N individually 
    peak_dist = np.argmin(np.abs(N-(x-x[0])))
    try:
        peak_idx = find_peaks(y,distance=peak_dist)[0] 
    except ValueError as e:
        print("Warning: ",e)
        peak_idx = find_peaks(y)[0] 
    # means Mi over each bin
    Mi = [np.mean(y[i_sta:i_end]) for i_sta,i_end in zip(peak_idx[:-1],peak_idx[1:])]
    # get mean of all bins and std
    N_bins = np.size(Mi)
    M_bins = np.mean(Mi)
    Std_bins = np.std(Mi)

    return Einf, Std_bins/np.sqrt(N_bins) + np.abs(Einf-M_bins), Mi, x
 