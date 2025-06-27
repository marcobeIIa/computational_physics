# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import scipy
import sys
import math
import scipy.special
import scipy.constants as const
from math import factorial
from scipy.special import hermite
import matplotlib.pyplot as plt
import random

print('tiocfaidh ar la v5')
# -

# In order to find the values of $\alpha$ that satisfy the cusp condition we notice that
# $$lim_{r_{ij}\rightarrow 0}\frac{H\Psi}{\Psi}<\infty$$
# where $H=H_0+V$ ,is equivalent to 
# $$lim_{r_{ij}\rightarrow 0}\frac{\Psi'}{\Psi}=\frac{1}{2}r_{ij}V$$
# having
# $$V=\frac{1}{2}\omega^2r_i^2+\frac{1}{r_{ij}}$$
# Substituting in this condition the Jastrow function for antiparallel spins and Jastrow function multiplied by $r$ fo parallel spins we find 
# $$\alpha_{\uparrow\downarrow}=\frac{1}{2}$$
# $$\alpha_{\uparrow\uparrow}=\frac{1}{4}$$
#
#
#
#
# Since we are interested in the cases up to $N=6$ let's consider only the lower eergy states in analitycal form, using the notation $\phi_{nlm}$
# $$\phi_{000}=\frac{1}{\sqrt{\pi\sigma^2}}e^{\frac{r^2}{2\sigma^2}}$$ 
# $$\phi_{01-1}=\frac{1}{\sqrt{\pi\sigma^2}}e^{\frac{r^2}{2\sigma^2}}\left(\frac{r}{\sigma}\right)e^{-i\varphi}$$ 
# $$\phi_{011}=\frac{1}{\sqrt{\pi\sigma^2}}e^{\frac{r^2}{2\sigma^2}}\left(\frac{r}{\sigma}\right)e^{i\varphi}$$ 
# Now we can write 
# $$e^{i\varphi}=cos\varphi+isin\varphi=\frac{x}{\sqrt{x^2+y^2}}+i\frac{y}{\sqrt{x^2+y^2}}$$
# $$e^{-i\varphi}=cos\varphi-isin\varphi=\frac{x}{\sqrt{x^2+y^2}}-i\frac{y}{\sqrt{x^2+y^2}}$$
# And so we find
# $$\phi_{000}=\frac{1}{\sqrt{\pi\sigma^2}}e^{\frac{x^2+y^2}{2\sigma^2}}$$ 
# $$\phi_{01-1}=\frac{1}{\sqrt{\pi\sigma^2}}e^{\frac{x^2+y^2}{2\sigma^2}}\left(\frac{\sqrt{x^2+y^2}}{\sigma}\right)\left(\frac{x}{\sqrt{x^2+y^2}}-i\frac{y}{\sqrt{x^2+y^2}}\right)=\frac{1}{\sqrt{\pi\sigma^2}}e^{\frac{x^2+y^2}{2\sigma^2}}\left(\frac{1}{\sigma}\right)\left(x-iy\right)$$ 
# $$\phi_{011}=\frac{1}{\sqrt{\pi\sigma^2}}e^{\frac{x^2+y^2}{2\sigma^2}}\left(\frac{\sqrt{x^2+y^2}}{\sigma}\right)\left(\frac{x}{\sqrt{x^2+y^2}}+i\frac{y}{\sqrt{x^2+y^2}}\right)=\frac{1}{\sqrt{\pi\sigma^2}}e^{\frac{x^2+y^2}{2\sigma^2}}\left(\frac{1}{\sigma}\right)\left(x+iy\right)$$ 
#
# In the end we can construct two new orbitals, namely
# $$\chi_+=\frac{\phi_{011}+\phi_{01-1}}{2}=\frac{1}{\sqrt{\pi\sigma^2}}e^{\frac{x^2+y^2}{2\sigma^2}}\frac{x}{\sigma}$$
# $$\chi_-=\frac{\phi_{011}-\phi_{01-1}}{2i}=\frac{1}{\sqrt{\pi\sigma^2}}e^{\frac{x^2+y^2}{2\sigma^2}}\frac{y}{\sigma}$$

N=6
omega=1


def Slaterdet(N,s,x1,y1,x2,y2,x3,y3,phi,chip,chim):
    A = [[(1/math.factorial(N))*phi(x1,y1,s), (1/math.factorial(N))*chip(x1,y1,s), (1/math.factorial(N))*chim(x1,y1,s)], 
         [(1/math.factorial(N))*phi(x2,y2,s), (1/math.factorial(N))*chip(x2,y2,s), (1/math.factorial(N))*chim(x2,y2,s)],
         [(1/math.factorial(N))*phi(x3,y3,s), (1/math.factorial(N))*chip(x3,y3,s), (1/math.factorial(N))*chim(x3,y3,s)]] 
    return np.linalg.det(A)



# +
def Slaterinv(N,s,x1,y1,x2,y2,x3,y3,phi,chip,chim):
    A = [[(1/math.factorial(N))*phi(x1,y1,s), (1/math.factorial(N))*chip(x1,y1,s), (1/math.factorial(N))*chim(x1,y1,s)], 
         [(1/math.factorial(N))*phi(x2,y2,s), (1/math.factorial(N))*chip(x2,y2,s), (1/math.factorial(N))*chim(x2,y2,s)],
         [(1/math.factorial(N))*phi(x3,y3,s), (1/math.factorial(N))*chip(x3,y3,s), (1/math.factorial(N))*chim(x3,y3,s)]] 
    
    return np.linalg.inv(A)

def safe_invert_matrix_mb(A, rcond=1e-15):
    """
    Computes a pseudoinverse if A is near-singular.

    Parameters:
    - A : (N x N) NumPy array
    - rcond : Regularization threshold

    Returns:
    - A_inv : Inverse or pseudoinverse of A
    """
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        print("Matrix is singular or ill-conditioned. Using pseudoinverse.")
        return np.linalg.pinv(A, rcond=rcond)



# +
#Metropolis
def Metropolis(x01,y01,x02,y02,x03,y03, delta, phi, chip, chim, s,counter, acc):
    a1=random.uniform(0,1)
    x1= x01+delta*(a1-0.5)
    b1=random.uniform(0,1)
    y1= y01+delta*(b1-0.5)
    a2=random.uniform(0,1)
    x2= x02+delta*(a2-0.5)
    b2=random.uniform(0,1)
    y2= y02+delta*(b2-0.5)
    a3=random.uniform(0,1)
    x3= x03+delta*(a3-0.5)
    b3=random.uniform(0,1)
    y3= y03+delta*(b3-0.5)
    p=Slaterdet(N,s,x1,y1,x2,y2,x3,y3,phi,chip,chim)**2/Slaterdet(N,s,x01,y01,x02,y02,x03,y03,phi,chip,chim)**2
    if p>1:
        x01=x1
        y01=y1
        x02=x2
        y02=y2
        x03=x3
        y03=y3
        acc+=1
    else:
        xi=random.uniform(0,1)
        if p>xi:
            x01=x1
            y01=y1
            x02=x2
            y02=y2
            x03=x3
            y03=y3
            acc+=1

    counter+=1
    return x01,y01,x02,y02,x03,y03, acc, counter



def adjust_delta(delta, acceptance_rate):
    if acceptance_rate > 0.6:
        delta *= 1.1  # Increase delta by 10%
    elif acceptance_rate < 0.4:
        delta *= 0.9  # Decrease delta by 10%
    return delta


def phi0(x, y, s):
    return (1/np.sqrt(np.pi*s**2))*math.exp((x**2+y**2)/(2*s**2))

def chiplus(x, y, s):
    return (1/np.sqrt(np.pi*s**2))*math.exp((x**2+y**2)/(2*s**2))*x/s

def chiminus(x, y, s):
    return (1/np.sqrt(np.pi*s**2))*math.exp((x**2+y**2)/(2*s**2))*y/s

def delta_choice(x01,y01,x02,y02,x03,y03,delta,s,N_delta,counter):
  acc_rate=1
  while acc_rate>0.6 or acc_rate<0.4:
    acc,counter=0,0
    for i in range(N_delta):
      x01,y01,x02,y02,x03,y03, acc,counter = Metropolis(x01,y01,x02,y02,x03,y03, delta, phi0, chiplus,chiminus, s,counter, acc)
    acc_rate=acc/counter
    delta = adjust_delta(delta, acc_rate)

  return delta

# + [markdown] magic_args="kinetic energy calculation"
# ## kinetic energy calculation
# Using
# \begin{align*}
#   \nabla ^2 (fg) = \left( \nabla ^2 f    \right) g + 2\boldsymbol{\nabla}f \cdot 
#   \boldsymbol{\nabla}g + f \nabla ^2g
# \end{align*}
#    The kinetic energy
# \begin{align}
#  \nabla ^2 \Psi &=  \nabla ^2 \left(F D_\uparrow D_\downarrow\right)
# \end{align}
# $$
#        \nabla^2 \Psi= \sum_i \left[   \underbrace{\left(\nabla _i ^2F\right)}_{\text{1}}D_\uparrow D_\downarrow +
#        2\underbrace{\boldsymbol{\nabla}_i F \cdot  \boldsymbol{\nabla}_i (D_\uparrow
#        D_\downarrow)}_{\text{2}} +
#        F(
#        \underbrace{(\nabla ^2_i D_\uparrow) D_\downarrow + D_\uparrow \nabla_i ^2
#        D_\downarrow)}_{\text{3}}\right]
# $$
# The first term
# \begin{align*}
#     (1) = \prod_{i<j}\left[\frac{-2a_{ij}b_{ij}}{(1+b_{ij}r_{ij})^{3}} +
#     \frac{a_{ij}^2}{(1+b_{ij}r_{ij})^{4}}+
#     \frac{1}{r_{ij}}\frac{a_{ij}}{(1+b_{ij}r_{ij})^2}\right]e^{\frac{a_{ij}r_{ij}}{1+b_{ij}r_{ij}}}
# \end{align*}
# then 
# \begin{align*}
#     \boldsymbol{\nabla} F \cdot \boldsymbol{\nabla} \left(D_\uparrow D_\downarrow\right)
#     &= 2\prod_{i\neq j}\hat{\textbf{x}}_{jk} \frac{a_{ij}}{(1+b_{ij}r_{ij})^2} \exp\{...\} \cdot
#     \boldsymbol{\nabla}(D_\uparrow D_\downarrow)
# \end{align*}
# and the gradient of Slater can be computed from known matrix calculus relations:
# \begin{align*}
# \frac{1}{\left|D\right|}\boldsymbol{\nabla}_i \left|D(\textbf{R})\right|    &= \sum _j
# \boldsymbol{\nabla}_i \phi_j (\textbf{r}_i) d_{ji}^{-1}(\textbf{R})\nonumber \\ 
# \frac{1}{\left|D\right|}\nabla ^2_i \left|D(\textbf{R})\right|    &= \sum _j
# \nabla^2_i \phi_j (\textbf{r}_i) d_{ji}^{-1}(\textbf{R})
# \end{align*}
# and the gradient of a single wavefunction
# \begin{align*}
#     \boldsymbol{\nabla}_i \phi_{000} (r_i)& = \frac{\textbf{r}_i}{\sigma^2}\phi_{000}\nonumber \\ 
# \boldsymbol{\nabla}_i \phi_{01 \pm } &= \begin{bmatrix} x_i + \sigma \\ y_i \pm i\sigma
# \end{bmatrix} \frac{1}{\sigma^2} \phi_{01 \pm }
# \end{align*}
# and the laplacians
# \begin{align*}
#    \nabla ^2 \phi_{000} &= \left(\frac{2r}{\sigma^2} +
#    \left(\frac{r}{\sigma^2}\right)^2\right)\phi_{000}\nonumber \\ 
#    \nabla ^2 \phi_{01 \pm }&= \frac{1}{\sqrt{\pi \sigma^2}} \frac{1}{\sigma^3}\left[2 +
#    \frac{x}{\sigma^2} \left(1+\frac{x}{\sigma^2}\right) + \frac{y}{\sigma^2}\left(\pm i +
#    \frac{y}{\sigma^2}\right)\right]e^{\frac{x^2+y^2}{2\sigma^2}}
# \end{align*}
#


# +
import importlib
import kinetic_energy as kin
import numpy as np
importlib.reload(kin)
# Parameters
N = 3
N_up = 2
sigma = 1
b_par = 1
b_orth = 1
omega = 1
use_chi = True

# Define a grid of positions (e.g., 3 particles in 2D)
R = np.array([
    [1.0, 3.0],
    [3.0, 2.0],
    [1.5, 0.266]  # roughly equilateral triangle
])
from functools import partial
phi0 =  partial(kin.single_particle_wf, m=0, sigma = sigma, use_chi = use_chi)
phi_plus =  partial(kin.single_particle_wf, m=1, sigma = sigma, use_chi = use_chi)
phi_minus =  partial(kin.single_particle_wf, m=-1, sigma = sigma, use_chi=use_chi)

print(kin.gradient_phi([0,0,0], R[0], sigma))


det_up = kin.slater_det(N_up,R[:N_up],phi0,phi_plus,phi_minus)
wavefunction = kin.total_wf(N, N_up, R, sigma, b_par, b_orth, use_chi = use_chi, return_A=False)[0]
partial_wf = partial(kin.total_wf,
                     N=N,
                     N_up=N_up,
                     sigma=sigma,
                     b_par=b_par,
                     b_orth=b_orth,
                     use_chi=True,
                     return_A=True)
f = lambda r: partial_wf(R=r)[0]
kinetic_energy_mine = kin.kinetic_energy_integrand(N, N_up,R, sigma, b_par, b_orth, omega, use_chi)
kinetic_energy_np = kin.numerical_laplacian_2D(f, R)
print("Kinetic energy (mine):", kinetic_energy_mine)
print("Kinetic energy (numpy):", kinetic_energy_np)
# -

acc, counter=0,0
x01=random.uniform(0,1)
y01=random.uniform(0,1)
x02=random.uniform(0,1)
y02=random.uniform(0,1)
x03=random.uniform(0,1)
y03=random.uniform(0,1)
delta=1
passi=10
Neq=int(passi*1/50)
N_delta=10000
pos, acc_list=[],[]
s=1

# +
delta = delta_choice(x01,y01,x02,y02,x03,y03,delta,s,N_delta,counter)

for i in range(passi):
    x01,y01,x02,y02,x03,y03, acc,counter = Metropolis(x01,y01,x02,y02,x03,y03, delta, phi0, chiplus,chiminus, s,counter, acc)
    #pos.append(x0)
    acc_list.append(acc/counter)

