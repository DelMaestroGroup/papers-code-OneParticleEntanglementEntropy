from mpmath import hyper
from mpmath import gamma as Γ
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi as π

kF = π/2

# relations between lattice model and LL model (from Bethe ansatz solution)


def γ_V(V):
    ''' Post quench Luttinger exponent  from ED interaction V
    '''
    return γ(KV(V))


def γ(K):
    '''Post quench Luttinger exponent.  
       see: https://link.aps.org/doi/10.1103/PhysRevA.80.063619 page 6, left column
    '''
    return 0.5*np.abs(K - 1.0/K)


def γeq_V(V):
    ''' Equilbrium LL exponent  from ED interaction V
    '''
    return γeq(KV(V))


def γeq(K):
    '''(Adrians function: quenchutils) Equilbrium LL exponent  
    '''
    return np.sqrt((K+1/K-2)/2)


def KV(V):
    ''''(Adrians function: quenchutils) Luttiger parameter  obtained from Bethe Ansatz of J-V model.
       see: http://link.aps.org/doi/10.1103/PhysRevLett.45.1358
       We take J = 1 here.
    '''
    return π/(2*np.arccos(-V/2))


def vV(V):
    ''''(Adrians function: quenchutils) Luttiger velocity (in units of J*latice spacing) obtained from Bethe Ansatz of J-V model.
       We take J = 1 here.
    '''
    return (1.0/(1 - np.arccos(-V/2)/π)) * np.sin(π*(1 - np.arccos(-V/2)/π))


def k_Fermi(L):
    if int(L/2) % 2:
        return kF
    else:
        return kF + π/L


def v_a_timeres(V):
    """Use the rescaling factor v/(t*a_0) with t=1, a_0=1."""
    return vV(V)

# analytic results for one body density


def fq_inf(qbar, γeq, ϵ):
    kFbar = ϵ*kF

    if np.abs(γeq) < 1e-14:
        if qbar**2 > kFbar**2:
            return 0
        return 1/(2*kF)

    c1 = Γ((γeq**2-1)/2) * np.sqrt(π) / (4*kF*π * Γ(γeq**2/2))
    c2 = 2 * Γ(-γeq**2)*np.sin(π*γeq**2/2) / (4*kF*π)

    def f1(qb):
        h1 = [1/2]
        h2 = [3/2, 3/2-γeq**2/2]
        h3 = 1/4 * (kFbar + qb)**2
        return (kFbar + qb) * hyper(h1, h2, h3)

    def f2(qb):
        h1 = [γeq**2/2]
        h2 = [(γeq**2+1)/2, 1+γeq**2/2]
        h3 = 1/4 * (kFbar + qb)**2
        return (kFbar + qb) * np.abs(kFbar + qb)**(γeq**2-1) * hyper(h1, h2, h3)

    res = c1 * (f1(qbar) + f1(-1.0*qbar)) - c2 * (f2(qbar) + f2(-1.0*qbar))

    return float(res.real)

def fq_fin_t(qbar,γ,ϵ,L,tres):
    q = qbar/ϵ
    def f_integrad(xi,q,γ,ϵ,L,tres):
        # We knwo that the result is real, so use only cos from np.exp(-1j*q*xi) 
        return rhoxi_fin_t(xi,γ,ϵ,L,tres) * np.cos(q*xi) 

    if int(L/2) % 2:
         norm = 1.0  
    else:
         norm = 1.0 #+ 2.0/L 
     

    return norm*quad(f_integrad, -0.5*L, 0.5*L, args=(q,γ,ϵ,L,tres))[0]

def rhoxi_fin_t(xi,γ,ϵ,L,tres): 

    if np.abs(xi) < 1e-13: 
        return 1/L
    
    _kF = kF #k_Fermi(L)

    c1 = π/(_kF*L)
    C0 = np.sin(_kF*xi)/(L*np.sin(π*xi/L))
    Cint = np.sin(π*1.0j*ϵ/L)/np.sin(π/L*(xi+1j*ϵ))
    Ct = np.sin(π/L*(xi-2*tres+1j*ϵ)) * np.sin(π/L*(xi+2*tres+1j*ϵ)) / np.sin(π/L*(2*tres+1j*ϵ)) / np.sin(π/L*(-2*tres+1j*ϵ))

    return c1 * C0 * np.abs(Cint)**(γ**2) * np.abs(Ct)**((γ**2)/2)  

# entanglement entropy


def Renyi_inf(fqbar, args, α, ϵ, limits):
    integral = quad(Renyi_integrand, *limits, args=(fqbar, args, α))[0]/ϵ

    if α == 1:
        return integral - np.log(2*kF)
    return 1/(1-α) * np.log(integral) - np.log(2*kF)


def Renyi_integrand(q, fq, args, α):
    if α == 1:
        return Nρlogρ(q, fq, args)
    if α == 1/2:
        return np.sqrt(np.abs(fq(q, *args)))

    return fq(q, *args)**α


def Nρlogρ(q, fq, args):
    '''N ρ(n=1) log ρ(n=1)
    '''
    f = fq(q, *args)
    if f < 1e-15:
        return 0.0

    return -f * np.log(np.abs(f))


def Renyi_fin(fq, args, α, ϵ, L, limits):
    integral = np.sum([Renyi_integrand(ϵ*2*π*n/L, fq, args, α)
                       for n in range(limits[0], limits[1]+1)])
    if α == 1:
        return integral
    return 1/(1-α) * np.log(integral)

def check_norm_fin(fq,args,ϵ,L,limits):
    return np.sum([fq(ϵ*2*π*n/L,*args) for n in range(limits[0],limits[1]+1)]) 

 