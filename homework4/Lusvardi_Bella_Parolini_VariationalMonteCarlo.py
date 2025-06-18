# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
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


# %% [markdown]
# In order to find the values of $\alpha$ that satisfy the cusp condition we notice that
# $$\lim_{r_{ij}\rightarrow 0}\frac{H\Psi}{\Psi}<\infty$$
# where $H=H_0+V$ ,is equivalent to 
# $$\lim_{r_{ij}\rightarrow 0}\frac{\Psi'}{\Psi}=\frac{1}{2}r_{ij}V$$
# having
# $$V=\frac{1}{2}\omega^2r_i^2+\frac{1}{r_{ij}}$$
# Substituting in this condition the Jastrow function for antiparallel spins and Jastrow function multiplied by $r$ fo parallel spins we find 
# $$\alpha_{\uparrow\downarrow}=\frac{1}{2}$$
# $$\alpha_{\uparrow\uparrow}=\frac{1}{4}$$
#
#
#
#
# Since we are interested in the cases up to $N=6$ let us consider only the lower energy states in analitycal form, using the notation $\phi_{nlm}$
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

# %%
N=6
omega=1


# %%
def Slaterdet(N,s,x1,y1,x2,y2,x3,y3,phi,chip,chim):
    A = [[(1/math.factorial(N))*phi(x1,y1,s), (1/math.factorial(N))*chip(x1,y1,s), (1/math.factorial(N))*chim(x1,y1,s)], 
         [(1/math.factorial(N))*phi(x2,y2,s), (1/math.factorial(N))*chip(x2,y2,s), (1/math.factorial(N))*chim(x2,y2,s)],
         [(1/math.factorial(N))*phi(x3,y3,s), (1/math.factorial(N))*chip(x3,y3,s), (1/math.factorial(N))*chim(x3,y3,s)]] 
    return np.linalg.det(A)


# %%
def Slaterinv(N,s,x1,y1,x2,y2,x3,y3,phi,chip,chim):
    A = [[(1/math.factorial(N))*phi(x1,y1,s), (1/math.factorial(N))*chip(x1,y1,s), (1/math.factorial(N))*chim(x1,y1,s)], 
         [(1/math.factorial(N))*phi(x2,y2,s), (1/math.factorial(N))*chip(x2,y2,s), (1/math.factorial(N))*chim(x2,y2,s)],
         [(1/math.factorial(N))*phi(x3,y3,s), (1/math.factorial(N))*chip(x3,y3,s), (1/math.factorial(N))*chim(x3,y3,s)]] 
    
    return np.linalg.inv(A)

# %%ctorial(N))*phi0(r[0],r[1],s), (1/math.factorial(N))*chip(r[0],r[1],s)], 
            [(1/math.factorial(N))*phi0(r[2],r[3],s), (1/math.factorial(N))*chip(r[2],r[3],s)]]
        return np.linalg.det(A)*phi0(r[4],r[6],s)
    elif N==4:
        A= [[(1/math.factorial(N))*phi0(r[0],r[1],s), (1/math.factorial(N))*chip(r[0],r[1],s)], 
            [(1/math.factorial(N))*phi0(r[2],r[3],s), (1/math.factorial(N))*chip(r[2],r[3],s)]]
        B= [[(1/math.factorial(N))*phi0(r[4],r[5],s), (1/math.factorial(N))*chip(r[4],r[5],s)], 
            [(1/math.factorial(N))*phi0(r[6],r[7],s), (1/math.factorial(N))*chip(r[6],r[7],s)]]
        return np.linalg.det(A)*np.linalg.det(B)
    elif N==5:
        A = [[(1/math.factorial(N))*phi0(r[0],r[1],s), (1/math.factorial(N))*chip(r[0],r[1],s), (1/math.factorial(N))*chim(r[0],r[1],s)], 
             [(1/math.factorial(N))*phi0(r[2],r[3],s), (1/math.factorial(N))*chip(r[2],r[3],s), (1/math.factorial(N))*chim(r[2],r[3],s)],
             [(1/math.factorial(N))*phi0(r[4],r[5],s), (1/math.factorial(N))*chip(r[4],r[5],s), (1/math.factorial(N))*chim(r[4],r[5],s)]]
        B= [[(1/math.factorial(N))*phi0(r[6],r[7],s), (1/math.factorial(N))*chip(r[6],r[7],s)], 
            [(1/math.factorial(N))*phi0(r[8],r[9],s), (1/math.factorial(N))*chip(r[8],r[9],s)]]
        return np.linalg.det(A)*np.linalg.det(B)
    elif N==6:
        A = [[(1/math.factorial(N))*phi0(r[0],r[1],s), (1/math.factorial(N))*chip(r[0],r[1],s), (1/math.factorial(N))*chim(r[0],r[1],s)], 
             [(1/math.factorial(N))*phi0(r[2],r[3],s), (1/math.factorial(N))*chip(r[2],r[3],s), (1/math.factorial(N))*chim(r[2],r[3],s)],
             [(1/math.factorial(N))*phi0(r[4],r[5],s), (1/math.factorial(N))*chip(r[4],r[5],s), (1/math.factorial(N))*chim(r[4],r[5],s)]]
        B = [[(1/math.factorial(N))*phi0(r[6],r[7],s), (1/math.factorial(N))*chip(r[6],r[7],s), (1/math.factorial(N))*chim(r[6],r[7],s)], 
             [(1/math.factorial(N))*phi0(r[8],r[9],s), (1/math.factorial(N))*chip(r[8],r[9],s), (1/math.factorial(N))*chim(r[8],r[9],s)],
             [(1/math.factorial(N))*phi0(r[10],r[11],s), (1/math.factorial(N))*chip(r[10],r[11],s), (1/math.factorial(N))*chim(r[10],r[11],s)]]
        return np.linalg.det(A)*np.linalg.det(B)
        


# %%
#Metropolis
def Metropolis(r,N, delta, psi, s,phi0,chip,chim,counter, acc):
    rn=r
    a=random.uniform(0,1)
    b=random.randint(0,2*N-1)
    rn[b]= r[b]+delta*(a-0.5)
    p=psi(phi0,chip,chim,N,r,s)**2/psi(phi0,chip,chim,N,r,s)**2
    if p>1:
        r[b]= rn[b]
        acc+=1
    else:
        xi=random.uniform(0,1)
        if p>xi:
            r[b]= rn[b]
            acc+=1

    counter+=1
    return r, acc, counter


# %%
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
def delta_choice(r,N,delta,s,N_delta,counter):
  acc_rate=1
  while acc_rate>0.6 or acc_rate<0.4:
    acc,counter=0,0
    for i in range(N_delta):
      x0, acc,counter = Metropolis(r,N, delta, psi, s,phi0,chiplus,chiminus,counter, acc)
    acc_rate=acc/counter
    delta = adjust_delta(delta, acc_rate)
    print(delta)

  return delta


# %%
acc, counter=0,0
x0=random.uniform(0,1)
delta=1
passi=1000
Neq=int(passi*1/50)
N_delta=1000
pos, acc_list=[],[]
s=1
r=np.zeros(12)
for i in range(len(r)):
    r[i]=random.uniform(0,7)
    
N=2

# %%
delta = delta_choice(r,N,delta,s,N_delta,0)


for i in range(passi):
    x0, acc, counter = Metropolis(r,N, delta, psi, s,phi0,chiplus,chiminus,counter, acc)
    pos.append(x0)
    acc_list.append(acc/counter)



# %%
plt.plot(np.array(range(passi)),acc_list)
plt.title('Acceptance rate')

# %%
