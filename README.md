# Diffusion Schr&ouml;dinger Bridge Matching

This repository contains the `PyTorch` implementation for the submission Diffusion Schr&ouml;dinger Bridge Matching.


## Introduction to Schr&ouml;dinger Bridges and links with diffusion models

The goal of learning Schr&ouml;dinger Bridges is to build a bridge between two distributions $\pi_ 0$ and $\pi_ T$ such that the bridge is optimal in some sense.
This transport setting covers many applications:
* Generative modeling: Gaussian $\rightarrow$ Data distribution.
* Data translation: Data distribution 1 $\rightarrow$ Data distribution 2.

The bridge is represented by a (stochastic) process $(\mathbf{X}_ t)_ {t \in [0,t]}$ such that $\mathbf{X}_ 0 \sim \pi_ 0$ and $\mathbf{X}_ T \sim \pi_ T$.

Schr&ouml;dinger bridges not only impose extremal constraints that the bridge must have the right distributions at time $0$ and $T$ but also imposes that the *energy* of the displacement is minimized in some sense.
As a result, Schr&ouml;dinger Bridges corresponds to solutions of (regularized) Optimal Transport problems.

Minimizing the energy of the path can also be interpreted at minimizing the Kullback-Leibler divergence between the measure of the bridge $\mathbb{P}$ and the measure of a *reference* process $\mathbb{Q}$ usually associated with a Brownian motion $(\mathbf{B}_ t)_ {t \in [0,T]}$. The Schr&ouml;dinger bridge $\mathbb{P}$ is solution to the following minimization problem.

$$
\mathbb{P}^\star = \arg\min \{ \mathrm{KL}(\mathbb{P}|\mathbb{Q}), \ \mathbb{P}_ 0 = \pi_0, \ \mathbb{P}_ T = \pi_T \} .
$$

The solution $\mathbb{P}^\star$ has the following properties:
1. $\mathbb{P}^\star_0 = \pi_0$.
2. $\mathbb{P}^\star_1 = \pi_T$.
3. $\mathbb{P}^\star$ is Markov.
4. $\mathbb{P}^\star$ is in the reciprocal class of $\mathbb{Q}$, i.e. $\mathbb{P}^\star_ {|0,T} = \mathbb{Q}_ {|0,T}$ (the measures $\mathbb{P}^\star$ and $\mathbb{Q}$ are the same when conditioned on the initial and terminal conditions).

The Iterative Proportional Fitting (IPF) procedure proceeds by alternatively projecting the measure on the conditions 1 and 2. The conditions 3 and 4 are satisfied for all the iterates. The new **Iterative Markovian Fitting** (IMF) procedure we propose alternatively projects on the condition 3 and 4, while preserving the conditions 1 and 2. 
We denote $\mathrm{proj}_ {\mathcal{M}}$ the projection on Markov processes and $\mathrm{proj}_ {\mathcal{R}(\mathbb{Q})}$ the projection on the reciprocal class of $\mathbb{Q}$.
The IMF procedure defines a sequence $(\mathbb{P}^n)_ {n \in \mathbb{N}}$ given by 

$$
\mathbb{P}^{2n+1} = \mathrm{proj}_ {\mathcal{M}}(\mathbb{P}^{2n}) , \\
\mathbb{P}^{2n+2} = \mathrm{proj}_ {\mathcal{R}(\mathbb{Q})}(\mathbb{P}^{2n+1}).    
$$

We refer to our paper for details on the implementation of these projections. The practical algorithm associated with IMF leverages Flow and Bridge Matching. We call this practical algorithm Diffusion Schr&ouml;dinger Bridge Matching (DSBM).

## Reproducing experiments
### Setting up
We provide a singularity container recipe in `bridge.def` which can be used to set up a singularity container. Alternatively, a conda environment can be set up manually using the conda installation commands in `bridge.def`.

### Gaussian experiment
A self-contained Gaussian experiment benchmark is provided in `DSBM-Gaussian.py`. 

DSB: `python DSBM-Gaussian.py dim=5,20,50 model_name=dsb seed=1,2,3,4,5 inner_iters=10000 -m`

IMF-b: `python DSBM-Gaussian.py dim=5,20,50 model_name=dsbm first_coupling=ind seed=1,2,3,4,5 inner_iters=10000 fb_sequence=['b'] -m`

DSBM-IPF: `python DSBM-Gaussian.py dim=5,20,50 model_name=dsbm seed=1,2,3,4,5 inner_iters=10000 -m`

DSBM-IMF: `python DSBM-Gaussian.py dim=5,20,50 model_name=dsbm first_coupling=ind seed=1,2,3,4,5 inner_iters=10000 -m`

Rectified Flow: `python DSBM-Gaussian.py dim=5,20,50 model_name=rectifiedflow seed=1,2,3,4,5 inner_iters=10000 fb_sequence=[b] -m`

SB-CFM: `python DSBM-Gaussian.py dim=5,20,50 model_name=sbcfm seed=1,2,3,4,5 inner_iters=10000 -m`


### MNIST experiment
DSBM-IPF: `python main.py num_steps=30 num_iter=5000 method=dbdsb gamma_min=0.034 gamma_max=0.034`

DSBM-IMF: `python main.py num_steps=30 num_iter=5000 method=dbdsb first_num_iter=100000 gamma_min=0.034 gamma_max=0.034 first_coupling=ind`

### Geophysical downscaling experiment
For the dataset, it can be downloaded and processed using the script `https://github.com/CliMA/diffusion-bridge-downscaling/blob/main/CliMAgen/examples/utils_data.jl`, then save as numpy arrays in `./data/downscaler`. 

DSBM-IPF: `python main.py dataset=downscaler_transfer num_steps=30 num_iter=5000 gamma_min=0.01 gamma_max=0.01 model=DownscalerUNET`

DSBM-IMF: `python main.py dataset=downscaler_transfer num_steps=30 num_iter=5000 gamma_min=0.01 gamma_max=0.01 model=DownscalerUNET first_coupling=ind`
