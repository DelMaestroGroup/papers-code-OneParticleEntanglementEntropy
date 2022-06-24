[![Paper](https://img.shields.io/badge/paper-arXiv%32206.11301-B31B1B.svg)](https://arxiv.org/abs/2206.11301)
[![DOI](https://zenodo.org/badge/504429972.svg)](https://zenodo.org/badge/latestdoi/504429972)


# One-particle entanglement for one dimensional spinless fermions after an interaction quantum quench

Matthias Thamm, Harini Radhakrishnan, Hatem Barghathi, Bernd Rosenow, and Adrian Del Maestro

[arXiv:2206.11301](https://arxiv.org/abs/2206.11301)

### Abstract
Particle entanglement provides information on quantum correlations in systems of indistinguishable particles. Here, we study the one particle entanglement entropy for an integrable model of spinless, interacting fermions both at equilibrium and after an interaction quantum quench. Using both large scale exact diagonalization and time dependent density matrix renormalization group calculations, we numerically compute the one body reduced density matrix for the $J$-$V$ model, as well as its post-quench dynamics.  We include an analysis of the fermionic momentum distribution, showcasing its time evolution after a quantum quench. Our numerical results, extrapolated to the thermodynamic limit, can be compared with field theoretic bosonization in the Tomonaga-Luttinger liquid regime. Excellent agreement is obtained using an interaction cutoff that can be determined uniquely in the ground state. 

### Description
This repository includes links, code, scripts, and data to generate the figures in the paper.

### Requirements
The data in this project was generated via exact diagonalization and dmrg.  
Everything included in the [data](https://github.com/DelMaestroGroup/papers-code-OneParticleEntanglementEntropy/tree/main/data) directory was generated via:

* [ED equilibrium code](https://github.com/DelMaestroGroup/tVDiagonalizeParticleEntanglementEntropyEquilibrium)
* [ED quench code](https://github.com/DelMaestroGroup/tVDiagonalizeParticleEntanglementEntropyQuench)
* [DMRG equilibrium and quench code](https://github.com/DelMaestroGroup/tVparticleEEdmrg_julia)

### Support
M.T. and B.R. acknowledge funding by the Deutsche Forschungsgemeinschaft (DFG) under Grant No. 406116891 within the Research Training Group RTG 2522/1 and under 
grant RO 2247/11-1. The creation of these materials was supported in part by the NSF under Grant No. DMR-2041995. 

[<img width="100px" src="https://www.nsf.gov/images/logos/NSF_4-Color_bitmap_Logo.png">](http://www.nsf.gov/awardsearch/showAward?AWD_ID=2041995)

### Figures

#### Figure 01: Interaction dependence of one particle von Neumann entanglement entropies
<img src="https://github.com/DelMaestroGroup/papers-code-OneParticleEntanglementEntropy/blob/main/figures/001_particleEntropy_overview.svg" width="400px"> 

#### Figure 02: Waiting time dependence of eigenvalues of one body density after quench
<img src="https://github.com/DelMaestroGroup/papers-code-OneParticleEntanglementEntropy/blob/main/figures/002_obdmspectrum_overview.svg" width="400px"> 

#### Figure 03: Distribution function after quantum quench
<img src="https://github.com/DelMaestroGroup/papers-code-OneParticleEntanglementEntropy/blob/main/figures/003_time_dependent_obdmf.svg" width="400px"> 

#### Figure 04: Distribution function from LL result
<img src="https://github.com/DelMaestroGroup/papers-code-OneParticleEntanglementEntropy/blob/main/figures/004_LL_analytic_fqInf.svg" width="400px"> 

#### Figure 05: Equilibrium one-particle entanglement entropy and phases of J-V model from DMRG results
<img src="https://github.com/DelMaestroGroup/papers-code-OneParticleEntanglementEntropy/blob/main/figures/005_stabilized_DMRG_eq_N51.svg" width="400px"> 

#### Figure 06: Finite size scaling of equilibrium one-particle entanglement entropy
<img src="https://github.com/DelMaestroGroup/papers-code-OneParticleEntanglementEntropy/blob/main/figures/006_equilibrium_finiteSizeScaling_posV_pow1.svg" width="400px"> 

#### Figure 07: Comparing numerical equilibrium one-particle entanglement to LL result in the thermodynamic limit
<img src="https://github.com/DelMaestroGroup/papers-code-OneParticleEntanglementEntropy/blob/main/figures/007_equilibrium_tdLimit_pow1.svg" width="400px"> 

#### Figure 08: Effective interaction cutoff in the equilibrium case
<img src="https://github.com/DelMaestroGroup/papers-code-OneParticleEntanglementEntropy/blob/main/figures/008_fittedCutoff.svg" width="400px"> 

#### Figure 09: Post quench LL steady state limit
<img src="https://github.com/DelMaestroGroup/papers-code-OneParticleEntanglementEntropy/blob/main/figures/009_quench_LL_plateau_scaling.svg" width="400px"> 

#### Figure 10: Evolution of one-particle entanglement after the quench
<img src="https://github.com/DelMaestroGroup/papers-code-OneParticleEntanglementEntropy/blob/main/figures/010_quench_timedependenceS1_N.svg" width="400px"> 

#### Figure 11: Finite size scaling of the steady state one-particle von Neumann entanglement entropy after the quench
<img src="https://github.com/DelMaestroGroup/papers-code-OneParticleEntanglementEntropy/blob/main/figures/011_quench_finiteSizeScaling_V_pow1.svg" width="400px"> 

#### Figure 12: Comparing numerical steady state one-particle entanglement to LL result in the thermodynamic limit after a quench
<img src="https://github.com/DelMaestroGroup/papers-code-OneParticleEntanglementEntropy/blob/main/figures/012_quench_tdLimit_pow1.svg" width="400px"> 

#### Figure 13: Interaction dependence of the effective cutoff
<img src="https://github.com/DelMaestroGroup/papers-code-OneParticleEntanglementEntropy/blob/main/figures/013_fittedCutoff_quench_pow1.svg" width="400px"> 

#### Figure 14: Time evolution of distribution function
<img src="https://github.com/DelMaestroGroup/papers-code-OneParticleEntanglementEntropy/blob/main/figures/014_obdmTimeEvolution.svg" width="700px"> 

#### Figure 15: Comparison time average of distribution function and equilibrium case
<img src="https://github.com/DelMaestroGroup/papers-code-OneParticleEntanglementEntropy/blob/main/figures/015_time_average_obdm.svg" width="400px"> 

#### Figure 16: Distribution function after quantum quench to CDW phase
<img src="https://github.com/DelMaestroGroup/papers-code-OneParticleEntanglementEntropy/blob/main/figures/016_time_dependent_obdmf_v20.svg" width="400px">   