import numpy as np
import sys
import matplotlib.pyplot as plt

# Parametri
E = 1.5 #meV
epsilon = 5.99 # meV
sigma = 2. # Angstrom
rmax = 150 # Angstrom
h = 0.01
N= rmax/h
r = np.arange(sigma/2, rmax, h)
l = 6

###############punto 1################
h_bar_c = 1.973e6 #mev ångström
m_H = 0.9383e12 #meV / c^2
m_Kr = 83.798 * m_H #meV / c^2
m_r = m_H * m_Kr / (m_H + m_Kr) #meV / c^2
k = E / h_bar_c #1/ångström WAVE number

prefactor = h_bar_c**2 / (2 * m_r) #meV ångström^2
######################################

# definire il potenziale
def V(r):
    V = 4*epsilon * ((sigma/r)**12 - (sigma/r)**6) #meV
    return V



# Metodo di Numerov
def Numerov(E,h,l,r):
    y = np.zeros(len(r))
    k = np.zeros(len(r))
    # Inizializzazione dei primi due valori di y e k
    y[0] = 0
    b = 1.25 #wtf is b?
    y[1] = np.exp(-(b/r[1])**5)
    k[0] = 1/prefactor * (E -  V(r[0])) - l * (l + 1) / r[0]**2
    k[1] = 1/prefactor * (E - V(r[1])) - l * (l + 1) / r[1]**2
    
    # Algoritmo di Numerov
    for j in range(2, len(r)):
        k[j] = 1/prefactor *(  E - V(r[j]) )- l * (l + 1) / r[j]**2
        y[j] = (1 / (1 + h**2 / 12 * k[j])) * (y[j-1] * (2 - 5 * h**2 / 6 * k[j-1]) - y[j-2] * (1 + h**2 / 12 * k[j-2]))

    psi = np.zeros(len(r))

    for j in range(len(r)):
        psi[j] = y[j] / r[j]
    return psi


# Normalizzazione
normalizzazione = 0       
n = np.zeros(len(r))

psi = Numerov(E,h,l,r)

for j in range(len(r)):
    n[j] = 4 * np.pi * psi[j]**2 * r[j]**2 *h
    normalizzazione += n[j]

PSI=psi/np.sqrt(normalizzazione)
# Visualizzazione
plt.plot(r,PSI)
plt.show()

#Bessel functions
def Bessel(r,l):
    j = np.zeros(l+1)
    n = np.zeros(l+1)
    j[0]=np.sin(r)/r
    n[0]=-np.cos(r)/r
    if l >= 1:
        j[1]=np.sin(r)/r**2-np.cos(r)/r
        n[1]=-np.cos(r)/r**2-np.sin(r)/r
    if l >= 2:
        for i in range(2,l+1):
            j[i]=((2*i-1)/r)*j[i-1]-j[i-2]

        for i in range(2,l+1):
            n[i]=((2*i-1)/r)*n[i-1]-n[i-2]
        
    return j,n
      
#initialise r1 and r2 for phase shift calculation
r1=len(r)-500
r2=len(r)-100


r1new=len(r)-3000
r2new=len(r)-5000

j_r1, n_r1 = Bessel(k*r[r1],l)
j_r2, n_r2 = Bessel(k*r[r2],l)

j_r1new, n_r1new = Bessel(k*r[r1new],l)
j_r2new, n_r2new = Bessel(k*r[r2new],l)


#phase shift
def Phase_shift(r1,r2,j_r1,j_r2,n_r1,n_r2,l):
    K=(PSI[r1]*r[r2])/((PSI[r2]*r[r1]))
    #delta_l=np.arctan(K*j_r2[l]-j_r1[l])/(K*n_r2[l]-n_r1[l])
    tan_delta_l=(K*j_r2[l]-j_r1[l])/(K*n_r2[l]-n_r1[l])
    return tan_delta_l

shift = Phase_shift(r1,r2,j_r1,j_r2,n_r1,n_r2,l)
shift2 = Phase_shift(r1new,r2new,j_r1new,j_r2new,n_r1new,n_r2new,l)
print(shift-shift2) 
print("the phase shit is ", shift) 
print("the phase shit2 is ", shift2) 
