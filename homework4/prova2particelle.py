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
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import scipy.special

# %%
jaud=1/2  #aij in Jastrow if antiparallel
jauu=1/4  #aij in Jastrow if parallel
jbud=1  #bij in Jastrow if antiparallel
jbuu=1  #bij in Jastrow if parallel

# %%
def Metropolisguess(r,delta):
    rn=np.zeros((len(r),len(r[0])))
    for i in range(len(r)):
        for j in range(len(r[0])):
            a=random.uniform(0,1)
            rn[i][j]= r[i][j]+delta*(a-0.5)
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

def chip(x,y,b):
    return x*psi(x,y,b)

def chim(x,y,b):
    return y*psi(x,y,b)

def psi2(r,b):
    return psi(r[0][0],r[1][0],b)*psi(r[0][1],r[1][1],b)

def psi3(r,b):
    A = [[psi(r[0][0],r[1][0],b),psi(r[0][1],r[1][1],b)], 
         [chip(r[0][0],r[1][0],b),chip(r[0][1],r[1][1],b)]]
    return np.linalg.det(A)*psi(r[0][2],r[1][2],b)
    
def psi4(r,b):
    A = [[psi(r[0][0],r[1][0],b),psi(r[0][1],r[1][1],b)], 
         [chip(r[0][0],r[1][0],b),chip(r[0][1],r[1][1],b)]]
    B = [[psi(r[0][2],r[1][2],b),psi(r[0][3],r[1][3],b)], 
         [chip(r[0][2],r[1][2],b),chip(r[0][3],r[1][3],b)]]
    return np.linalg.det(A)*np.linalg.det(B)

def psi5(r,b):
    A = [[psi(r[0][0],r[1][0],b),psi(r[0][1],r[1][1],b),psi(r[0][2],r[1][2],b)], 
         [chip(r[0][0],r[1][0],b),chip(r[0][1],r[1][1],b),chip(r[0][2],r[1][2],b)],
         [chim(r[0][0],r[1][0],b),chim(r[0][1],r[1][1],b),chim(r[0][2],r[1][2],b)]]
    B = [[psi(r[0][3],r[1][3],b),psi(r[0][4],r[1][4],b)], 
         [chip(r[0][3],r[1][3],b),chip(r[0][4],r[1][4],b)]]
    return np.linalg.det(A)*np.linalg.det(B)

def psi6(r,b):
    A = [[psi(r[0][0],r[1][0],b),psi(r[0][1],r[1][1],b),psi(r[0][2],r[1][2],b)], 
         [chip(r[0][0],r[1][0],b),chip(r[0][1],r[1][1],b),chip(r[0][2],r[1][2],b)],
         [chim(r[0][0],r[1][0],b),chim(r[0][1],r[1][1],b),chim(r[0][2],r[1][2],b)]]
    B = [[psi(r[0][3],r[1][3],b),psi(r[0][4],r[1][4],b),psi(r[0][5],r[1][5],b)], 
         [chip(r[0][3],r[1][3],b),chip(r[0][4],r[1][4],b),chip(r[0][5],r[1][5],b)],
         [chim(r[0][3],r[1][3],b),chim(r[0][4],r[1][4],b),chim(r[0][5],r[1][5],b)]]
    return np.linalg.det(A)*np.linalg.det(B)

def jastrowfunct(r,a,b):
    return math.exp(a*r/(1+b*r))

# %%
def Jastrow(r,a,b):
    j=1
    for i in range(len(r)):
        j*=jastrowfunct(r[i],a,b)
    return j

# %%
def Jastrowpsi(r,b):
    return psix(r[0][0],b)*psiy(r[1][0],b)

def Jastrowpsi2(r,b):
     part=2
     xij=np.zeros(int(part*(part-1)/2))
     yij=np.zeros(int(part*(part-1)/2))
     k=0
     for i in range(len(r[0])):
          for j in range(i):
               xij[k]=r[0][i]-r[0][j]
               yij[k]=r[1][i]-r[1][j]
               k+=1

     rij=np.zeros(int(part*(part-1)/2))
     for i in range(len(rij)):
          rij[i]=np.sqrt(xij[i]**2+yij[i]**2)
          
          
     return psi(r[0][0],r[1][0],b)*psi(r[0][1],r[1][1],b)*Jastrow(rij,jauu,jbuu)

def Jastrowpsi3(r,b):
     part=3
     xij=np.zeros(int(part*(part-1)/2))
     yij=np.zeros(int(part*(part-1)/2))
     k=0
     for i in range(len(r[0])):
          for j in range(i):
               xij[k]=r[0][i]-r[0][j]
               yij[k]=r[1][i]-r[1][j]
               k+=1

     rij=np.zeros(int(part*(part-1)/2))
     for i in range(len(rij)):
          rij[i]=np.sqrt(xij[i]**2+yij[i]**2)
     
     A = [[psi(r[0][0],r[1][0],b),psi(r[0][1],r[1][1],b)], 
         [chip(r[0][0],r[1][0],b),chip(r[0][1],r[1][1],b)]]
     
     return np.linalg.det(A)*psi(r[0][2],r[1][2],b)*Jastrow(rij,jauu,jbuu)
    
def Jastrowpsi4(r,b):
     part=4
     xij=np.zeros(int(part*(part-1)/2))
     yij=np.zeros(int(part*(part-1)/2))
     k=0
     for i in range(len(r[0])):
          for j in range(i):
               xij[k]=r[0][i]-r[0][j]
               yij[k]=r[1][i]-r[1][j]
               k+=1

     rij=np.zeros(int(part*(part-1)/2))
     for i in range(len(rij)):
          rij[i]=np.sqrt(xij[i]**2+yij[i]**2)
     A = [[psi(r[0][0],r[1][0],b),psi(r[0][1],r[1][1],b)], 
         [chip(r[0][0],r[1][0],b),chip(r[0][1],r[1][1],b)]]
     B = [[psi(r[0][2],r[1][2],b),psi(r[0][3],r[1][3],b)], 
         [chip(r[0][2],r[1][2],b),chip(r[0][3],r[1][3],b)]]
     return np.linalg.det(A)*np.linalg.det(B)*Jastrow(rij,jauu,jbuu)

def Jastrowpsi5(r,b):
     part=5
     xij=np.zeros(int(part*(part-1)/2))
     yij=np.zeros(int(part*(part-1)/2))
     k=0
     for i in range(len(r[0])):
          for j in range(i):
               xij[k]=r[0][i]-r[0][j]
               yij[k]=r[1][i]-r[1][j]
               k+=1

     rij=np.zeros(int(part*(part-1)/2))
     for i in range(len(rij)):
          rij[i]=np.sqrt(xij[i]**2+yij[i]**2)
     A = [[psi(r[0][0],r[1][0],b),psi(r[0][1],r[1][1],b),psi(r[0][2],r[1][2],b)], 
         [chip(r[0][0],r[1][0],b),chip(r[0][1],r[1][1],b),chip(r[0][2],r[1][2],b)],
         [chim(r[0][0],r[1][0],b),chim(r[0][1],r[1][1],b),chim(r[0][2],r[1][2],b)]]
     B = [[psi(r[0][3],r[1][3],b),psi(r[0][4],r[1][4],b)], 
         [chip(r[0][3],r[1][3],b),chip(r[0][4],r[1][4],b)]]
     return np.linalg.det(A)*np.linalg.det(B)*Jastrow(rij,jauu,jbuu)

def Jastrowpsi6(r,b):
     part=6
     xij=np.zeros(int(part*(part-1)/2))
     yij=np.zeros(int(part*(part-1)/2))
     k=0
     for i in range(len(r[0])):
          for j in range(i):
               xij[k]=r[0][i]-r[0][j]
               yij[k]=r[1][i]-r[1][j]
               k+=1

     rij=np.zeros(int(part*(part-1)/2))
     for i in range(len(rij)):
          rij[i]=np.sqrt(xij[i]**2+yij[i]**2)
     A = [[psi(r[0][0],r[1][0],b),psi(r[0][1],r[1][1],b),psi(r[0][2],r[1][2],b)], 
         [chip(r[0][0],r[1][0],b),chip(r[0][1],r[1][1],b),chip(r[0][2],r[1][2],b)],
         [chim(r[0][0],r[1][0],b),chim(r[0][1],r[1][1],b),chim(r[0][2],r[1][2],b)]]
     B = [[psi(r[0][3],r[1][3],b),psi(r[0][4],r[1][4],b),psi(r[0][5],r[1][5],b)], 
         [chip(r[0][3],r[1][3],b),chip(r[0][4],r[1][4],b),chip(r[0][5],r[1][5],b)],
         [chim(r[0][3],r[1][3],b),chim(r[0][4],r[1][4],b),chim(r[0][5],r[1][5],b)]]
     return np.linalg.det(A)*np.linalg.det(B)*Jastrow(rij,jauu,jbuu)

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
    elif n==5:
        phi = psi5
    elif n==6:
        phi = psi6
    return phi


# %%
def functchoicejastrow(n):
    if n==1:
        phi = Jastrowpsi
    elif n==2:
        phi = Jastrowpsi2
    elif n==3:
        phi = Jastrowpsi3
    elif n==4:
        phi = Jastrowpsi4
    elif n==5:
        phi = Jastrowpsi5
    elif n==6:
        phi = Jastrowpsi6
    return phi

# %% [markdown]
# ### POINT 4

# %%
b=1

part = 2
funct=functchoice(part)
r = np.zeros((2,part))
acc, counter=0,0
for i in range(len(r)):
    for j in range(len(r[0])):
        r[i][j]=random.uniform(0,1)   #r[x o y][particella]
        

delta=1
N=1000000
Neq=int(N*1/50)
N_delta=10000
posx1,posy1,posx2,posy2, acc_list=[],[],[],[],[]


# %%
delta = delta_choice(r,delta,b,N_delta,0,funct)

for i in range(N):
    r,acc, counter = Metropolis(r, delta, funct, b,counter, acc)
    
    posx1.append(r)

    acc_list.append(acc/counter)


print(f'b={b},\u0394={round(delta,3)}')


# %%
plt.plot(np.array(range(N)),acc_list)
plt.title('Acceptance rate')

# %% [markdown]
# ### POINT 5

# %%
b=1



part = 3

r = np.zeros((2,part))
acc, counter=0,0
for i in range(len(r)):
    for j in range(len(r[0])):
        r[i][j]=random.uniform(0,1)   #r[x o y][particella]
        

funct=functchoicejastrow(part)



acc, counter=0,0
delta=1
N=1000000
Neq=int(N*1/50)
N_delta=10000
posx1,posy1,posx2,posy2, acc_list=[],[],[],[],[]

# %%
delta = delta_choice(r,delta,b,N_delta,0,funct)

for i in range(N):
    r,acc, counter = Metropolis(r, delta, funct, b,counter, acc)
    
    posx1.append(r)

    acc_list.append(acc/counter)


print(f'b={b},\u0394={round(delta,3)}')


# %%
plt.plot(np.array(range(N)),acc_list)
plt.title('Acceptance rate')


