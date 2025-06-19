# %%
import random
import numpy as np
import math
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

import matplotlib.pyplot as plt

# %%
def Metropolisguess(r,delta):
    rn=np.zeros(len(r))
    for i in range(len(r)):
        a=random.uniform(0,1)
        rn[i]= r[i]+delta*(a-0.5)
    return rn

# %%
#Metropolis
def Metropolis(r,delta, psi, b,counter, acc):
    rn = Metropolisguess(r,delta)
    p=psi(rn,b)**2/psi(r,b)**2
    if p>1:
        r=rn
        acc+=1
    else:
        xi=random.uniform(0,1)
        if p>xi:
            r=rn
            acc+=1

    counter+=1
    return r,acc, counter

# %%
def adjust_delta(delta, acceptance_rate):
    if acceptance_rate > 0.6:
        delta *= 1.1  # Increase delta by 10%
    elif acceptance_rate < 0.4:
        delta *= 0.9  # Decrease delta by 10%
    return delta

# %%
def psix(x, b):
    return math.exp(-x**2/(2*b**2))

def psiy(y, b):
    return math.exp(-y**2/(2*b**2))

def psi(x,y,b):
    return psix(x,b)*psiy(y,b)

def psi2(r,b):
    return psi(r[0],r[1],b)*psi(r[2],r[3],b)

def psi3(r,b):
    return (psi(r[0],r[1],b)*r[2]*psi(r[2],r[3],b)-psi(r[2],r[3],b)*r[0]*psi(r[0],r[1],b))*psi(r[4],r[5],b)

def psi4(r,b):
    return (psi(r[0],r[1],b)*r[2]*psi(r[2],r[3],b)-psi(r[2],r[3],b)*r[0]*psi(r[0],r[1],b))*(psi(r[4],r[5],b)*r[2]*psi(r[6],r[7],b)-psi(r[6],r[7],b)*r[0]*psi(r[4],r[5],b))

# %%
def delta_choice(r,delta,b,N_delta,counter,psi):
  acc_rate=1
  while acc_rate>0.6 or acc_rate<0.4:
    acc,counter=0,0
    for i in range(N_delta):
      r, acc,counter = Metropolis(r, delta, psi, b,counter, acc)
    acc_rate=acc/counter
    delta = adjust_delta(delta, acc_rate)

  return delta

# %%
def functchoice(n):
    if n==1:
        phi = psi
    elif n==2:
        phi = psi2
    elif n==3:
        phi = psi3
    elif n==4:
        phi = psi4
    return phi
        

# %%
b=1

part = 2
funct=functchoice(part)
r = np.zeros(2*part)
acc, counter=0,0
for i in range(len(r)):
    r[i]=random.uniform(0,1)
delta=1
N=1000000
Neq=int(N*1/50)
N_delta=10000
posx1,posy1,posx2,posy2, acc_list=[],[],[],[],[]
sum_E, Esq, s1, s2,s1sq,s2sq= 0,0,0,0,0,0

# %%
delta = delta_choice(r,delta,b,N_delta,0,funct)

for i in range(N):
    r,acc, counter = Metropolis(r, delta, funct, b,counter, acc)
    posx1.append(r[0])
    posy1.append(r[1])
    posx2.append(r[2])
    posy2.append(r[3])
    acc_list.append(acc/counter)


print(f'b={b},\u0394={round(delta,3)}')


# %%
plt.plot(np.array(range(N)),acc_list)
plt.title('Acceptance rate')

# %%
y=np.arange(0,N,1)
plt.plot(y,posx1, label='x1')
plt.plot(y,posy1, label='y1')

# %%
y=np.arange(0,N,1)
plt.plot(y,posx2, label='x2')
plt.plot(y,posy2, label='y2')


