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
# ## part 2: calculation of the kinetic energy
#
# Using
# \begin{align*}
#   \nabla ^2 (fg) = \left( \nabla ^2 f    \right) g + 2\boldsymbol{\nabla}f \cdot 
#   \boldsymbol{\nabla}g + f \nabla ^2g
# \end{align*}
#    The kinetic energy
# \begin{align*}
#        \nabla ^2 \Psi &=  \nabla ^2 \left(J D_\uparrow D_\downarrow\right)\nonumber \\ 
#        &= \sum_i \left[   \left(\nabla _i ^2J\right)D_\uparrow D_\downarrow +
#        2\boldsymbol{\nabla}_i J \cdot  \boldsymbol{\nabla}_i (D_\uparrow
#        D_\downarrow) +
#        J((\nabla ^2_i D_\uparrow) D_\downarrow + D_\uparrow \nabla_i ^2
#        D_\downarrow)\right]
# \end{align*}
# The first term is computed as
# \begin{align*}
#     \nabla ^2_i J &= \nabla ^2 _i \prod_{j<k} e^{\log f_{jk}}\nonumber \\ 
#     &=\nabla_i ^2 \exp \sum_{j < k} f_{jk} = \nabla _i ^2 \exp \left(\frac{1}{2}\sum_{i\neq
#     k}f_{jk}\right)
# \end{align*}
# We exploit the identity
# \begin{align*}
#     \nabla^2 f(g(\textbf{x})) &= f^{\prime\prime} (g(\textbf{x}))
#    \left|\boldsymbol{\nabla}g(x)\right|^2  + f^\prime (g(\textbf{x})) \nabla ^2 g.
# \end{align*}
# (Skipping a few lines of algebra), I find
# \begin{align*}
#     \nabla _i ^2 J &= J \left(\left|\sum_{k\neq i}\frac{f^\prime
#     _{ki}}{f_{ki}}\hat{\textbf{x}}_{jk}\right|^2 + \sum_{k\neq i}\nabla _i ^2 \log f_{ki}\right)
# \end{align*}
# where
# \begin{align*}
#     \nabla _i ^2 \log f_{ki} &= \frac{f^{\prime\prime}_{ki}}{f_{ki}} -
#     \frac{(f_{ki}^\prime )^2}{f_{ki}^2} + \frac{1}{r_{ki}} \frac{f_{ki}^\prime }{f_{ki}}.
# \end{align*}
#
# Then 
# \begin{align*}
#     \boldsymbol{\nabla} J \cdot \boldsymbol{\nabla} \left(D_\uparrow D_\downarrow\right)
#     &= \sum_{i} \left(\nabla _i J\right)\nabla _i\left(D_\uparrow D_\downarrow\right).
# \end{align*}
# The gradient of Jastrow is
# \begin{align*}
#     \nabla _i J &= J \nabla _i \log J\nonumber \\ 
#     &= \frac{1}{2}J \nabla _i \left(\sum_{jk,j\neq k} \ln f_{jk}\right)\nonumber \\ 
#     &= J\sum_{k,k\neq i} \frac{f_{ik}^\prime }{f_{ik}} \hat{\textbf{r}}_{ik}
# \end{align*}
# where the $1 /2$ (coming from  $2\sum_{j<k} = \sum_{j\neq k}$) is reabsorbed since we get
# two equal terms: one from $i=j$ and one from $i=k$.
#
# The gradient of Slater can be computed from known matrix calculus relations:
# \begin{align*}
# \frac{1}{\left|D\right|}\boldsymbol{\nabla}_i \left|D(\textbf{R})\right|    &= \sum _j
# \boldsymbol{\nabla}_i \phi_j (\textbf{r}_i) d_{ji}^{-1}(\textbf{R})
# \end{align*}
# and the gradient of a single wavefunction
# \begin{align*}
#     \boldsymbol{\nabla}_i \phi_{000} (r_i)& =
#     -\frac{\textbf{r}_i}{\sigma^2}\phi_{000}\nonumber \\
#     \boldsymbol{\nabla}_i \chi_{+} &= \begin{bmatrix} 1- x_{i}^2/\sigma^2 \\ -x_i y_i /
#     \sigma^2
# \\\end{bmatrix} \frac{1}{\sigma^2 \sqrt{\pi} } e^{- (x_{i}^2 +y_{i}^2) / 2 \sigma^2}\nonumber \\ 
#     \boldsymbol{\nabla}_i \chi_{-} &= \begin{bmatrix} -x y/\sigma^2 \\ 1 -y_i ^2 /
#     \sigma^2
# \\\end{bmatrix} \frac{1}{\sigma^2 \sqrt{\pi} } e^{- (x^2 +y^2) / 2 \sigma^2}.
# \end{align*}
# Finally for the Slater laplacian term, we employ a similar result:
# \begin{align*}
# \frac{1}{\left|D\right|}\nabla ^2_i \left|D(\textbf{R})\right|    &= \sum _j
# \nabla^2_i \phi_j (\textbf{r}_i) d_{ji}^{-1}(\textbf{R})
# \end{align*}
# where the single-particle laplacians are found via the equations of motion
# \begin{align*}
#   \left(  -\frac{1}{2}\nabla_i ^2 +\frac{1}{2}\omega^2 r_i^2\right) \phi_j (r_i) =
#   \lambda_j \phi_j
#   (r_i)
# \end{align*}
# or 
# \begin{align*}
#     \nabla_i ^2 \phi_j (r_i) = \left(\omega^2 r_i^2 - 2\lambda_j\right)\phi_j(r_i)
# \end{align*}
# for 
# \begin{align*}
#     \lambda_j = \omega (2n + l + 1).
# \end{align*}
# These calculations, put together, give us the first way of computing the integrand of the
# kinetic energy.
#
# -
# ### sample run
#
# note that the integrand is lacking the 
# \begin{align}
# -\frac{\hbar ^2}{2m}
# \end{align}
# prefactor ;)

# +
import libraryMetropolis as lib

N_test = 5
R_test = np.random.uniform(0,2,(N_test,2))
b_par = 1 
b_anti = 1
sigma = 1

K_integrand = lib.kinetic_energy_integrand(N_test,lib.N_up_choice(N_test),R_test,sigma,b_par,b_anti,omega)
print(K_integrand)
# -
#

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

