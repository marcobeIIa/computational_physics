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
# $$\chi_+=\frac{\phi_{011}+\phi_{01-1}}{2}=\frac{1}{\sqrt{\pi\sigma^2}}e^{\frac{x^2+y^2}{2\sigma^2}}x$$
# $$\chi_-=\frac{\phi_{011}-\phi_{01-1}}{2i}=\frac{1}{\sqrt{\pi\sigma^2}}e^{\frac{x^2+y^2}{2\sigma^2}}y$$

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
    return (1/np.sqrt(np.pi*s**2))*math.exp((x**2+y**2)/(2*s**2))*x

def chiminus(x, y, s):
    return (1/np.sqrt(np.pi*s**2))*math.exp((x**2+y**2)/(2*s**2))*y

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
# -


# +
# here i will define all functions to take as input
# N = number of particles
# R = [[x1, y2],...,[xN, yN]]
# A = [[a11,...,a12],...,[aN1,..., aNN]]
# B = [[b11,...,b12],...,[bN1,..., bNN]]

def A_matrix_creator_mb(N, R, sigma, phi, chip, chim):
    """
    creates the matrix from which we compute the slater determinant

    Parameters:
    - N      : Number of particles
    - R      : Array of shape (N, 2), each row is [x, y] of particle
    - sigma  : Sigma parameter in h.o. wavefunction
    - phi    : Callable: phi(x, y, spin)
    - chip   : Callable: chip(x, y, spin)
    - chim   : Callable: chim(x, y, spin)

    returns :
    - A[i,j] = phi_j(x_i, y_i, spin_i) 
    """
    basis_functions = [phi, chip, chim]
    num_basis = len(basis_functions)
    if N > num_basis:
        raise ValueError("not enough basis functions for n particles")

    A = np.zeros((N, N), dtype=float)

    for i in range(N):  # row: particle
        x, y = R[i]
        for j in range(N):  # col: basis function
            A[i, j] = basis_functions[j](x, y, sigma)

def slater_det_mb(N, R, sigma, phi, chip, chim, normalised = False):
    """
    Compute the Slater determinant for N particles.

    Parameters:
    - N      : Number of particles
    - R      : Array of shape (N, 2), each row is [x, y] of particle
    - sigma  : Sigma parameter for the harmonic oscillator wavefunction
    - phi    : Callable: phi(x, y, spin)
    - chip   : Callable: chip(x, y, spin)
    - chim   : Callable: chim(x, y, spin)

    Returns:
    - det    : Value of the Slater determinant
    """
    basis_functions = [phi, chip, chim]
    num_basis = len(basis_functions)

    if N > num_basis:
        raise ValueError("not enough basis functions for N particles")

    A = A_matrix_creator(N, R, sigma, phi, chip, chim)

    # Normalization factor, i dont wanna normalise because im scared of big numbers
    if not normalised:
        normalisation = 1
    else: 
        normalisation = 1 / math.factorial(N)
    return normalisation * np.linalg.det(A)


def jastrow_laplacian_mb(N,R,A,B):
    '''
    This function calculates the laplacian of jastrow, or (1)
    '''
    out = 1
    for i in range(N):
        for j in range(i+1,N):
            rij = np.linalg.norm(R[i] - R[j])
            aij = A[i][j]
            bij = B[i][j]
            x = 1+bij*rij
            out *= (-2*aij*bij/ x + aij**2/x**2 + aij/rij)/x**2 * np.exp(-aij*rij / x)
    return out

def gradient_phi_mb(alpha, r,sigma):
    '''
    alpha =  [n,l,m] orbital nmbers
    r= [x,y] = coordinates of i-th particle
    sigma the usual
    '''
    x,y = r
    n,l,m = alpha
    factor = np.exp((x**2+y**2)/(2*alpha[0]**2))/ sigma**2
    if alpha == [0,0,0]:
        return np.array(x* factor,
                         y* factor)
    elif [n,l] == [0,1]:
        return np.array((x+sigma)* factor,
                         (y+1j*m*sigma)* factor)

def gradient_chi_mb(m, r,sigma):
    if m == 1:
        return (gradient_phi_mb([0,1,1], r, sigma) + gradient_phi_mb([0,-1,1], r, sigma))/2
    elif m ==-1:
        return (gradient_phi_mb([0,1,1], r, sigma) - gradient_phi_mb([0,-1,1], r, sigma))/(2j)
    else:
        print("Invalid value for m = +-1")


def slater_gradient_mb(N,R,A_inv,i,det,sigma):
    ''' 
    This function calculates the gradient of the Slater determinant
    A_inv - the inverse of the Slater matrix
    i - i-th partiche wrt which we're computing the gradient 
    det - the Slater determinant itself
    '''
    out = np.zeros((1, 2))
    alpha = [[0, 0, 0], [0, 1, 1], [0, 1, -1]]  # Assuming three basis functions
    for j in range(N):
        out += A_inv[i][j] * gradient_phi_mb(alpha[j], R[i], sigma)


def gradient_gradient_term_mb(N,R,A,B):
    '''
    This function calculates the gradient gradient term of jastrow, or (2)
    '''
    out = 0
    for i in range(N):
        for j in range(i+1,N):
            rij = np.linalg.norm(R[i] - R[j])
            aij = A[i][j]
            bij = B[i][j]
            x = 1+bij*rij
            jastrow_prefactor = aij/ x**2 * np.exp(-aij*rij / x)
            jastrow_piece = jastrow_prefactor* (R[i]-R[j]) #this guy should be a vector
    return out


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

