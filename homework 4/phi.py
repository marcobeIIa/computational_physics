import math
import numpy as np

jaud=1/2  #aij in Jastrow if antiparallel
jauu=1/4  #aij in Jastrow if parallel
jbud=1  #bij in Jastrow if antiparallel
jbuu=1  #bij in Jastrow if parallel

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

def single_wf_choice(n):
    if n==0:
        return psi
    if n==1:
        return chip
    if n==-1:
        return chim


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

def Jastrow(r,aligned):
    j=1
    for i in range(len(r)):
        if aligned[i] == True:
            a = jauu
            b = jbuu
        else:
            a = jaud
            b = jbud
        j*=jastrowfunct(r[i],a,b)
    return j

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
#     print("Lusva rij", rij)
     return psi(r[0][0],r[1][0],b)*psi(r[0][1],r[1][1],b)*Jastrow(rij,[False])

def Jastrowpsi3(r,b):
     part=3
     xij=np.zeros(int(part*(part-1)/2))
     yij=np.zeros(int(part*(part-1)/2))
     k=0
     aligned = np.zeros(int(part * (part - 1) / 2), dtype=bool)
     for i in range(len(r[0])):
          for j in range(i):
               xij[k]=r[0][i]-r[0][j]
               yij[k]=r[1][i]-r[1][j]
               if i < 2 and j <2 or i >= 2 and j>=2:
                   aligned[k] = True
               else:
                   aligned[k] = False
               k+=1
     rij=np.zeros(int(part*(part-1)/2))
     for i in range(len(rij)):
          rij[i]=np.sqrt(xij[i]**2+yij[i]**2)
     
     A = [[psi(r[0][0],r[1][0],b),psi(r[0][1],r[1][1],b)], 
         [chip(r[0][0],r[1][0],b),chip(r[0][1],r[1][1],b)]]
     
     return np.linalg.det(A)*psi(r[0][2],r[1][2],b)*Jastrow(rij,aligned)
    
def Jastrowpsi4(r,b):
     part=4
     xij=np.zeros(int(part*(part-1)/2))
     yij=np.zeros(int(part*(part-1)/2))
     k=0
     aligned = np.zeros(int(part * (part - 1) / 2), dtype=bool)
     for i in range(len(r[0])):
          for j in range(i):
               xij[k]=r[0][i]-r[0][j]
               yij[k]=r[1][i]-r[1][j]
               if i < 2 and j <2 or i >= 2 and j>=2:
                   aligned[k] = True
               else:
                   aligned[k] = False
               k+=1

     rij=np.zeros(int(part*(part-1)/2))
     for i in range(len(rij)):
          rij[i]=np.sqrt(xij[i]**2+yij[i]**2)
     A = [[psi(r[0][0],r[1][0],b),psi(r[0][1],r[1][1],b)], 
         [chip(r[0][0],r[1][0],b),chip(r[0][1],r[1][1],b)]]
     B = [[psi(r[0][2],r[1][2],b),psi(r[0][3],r[1][3],b)], 
         [chip(r[0][2],r[1][2],b),chip(r[0][3],r[1][3],b)]]
     return np.linalg.det(A)*np.linalg.det(B)*Jastrow(rij,aligned)

def Jastrowpsi5(r,b):
     part=5
     xij=np.zeros(int(part*(part-1)/2))
     yij=np.zeros(int(part*(part-1)/2))
     k=0
     aligned = np.zeros(int(part * (part - 1) / 2), dtype=bool)
     for i in range(len(r[0])):
          for j in range(i):
               xij[k]=r[0][i]-r[0][j]
               yij[k]=r[1][i]-r[1][j]
               if i < 3 and j <3 or i >= 3 and j>=3:
                   aligned[k] = True
               else:
                   aligned[k] = False
               k+=1

     rij=np.zeros(int(part*(part-1)/2))
     for i in range(len(rij)):
          rij[i]=np.sqrt(xij[i]**2+yij[i]**2)
     A = [[psi(r[0][0],r[1][0],b),psi(r[0][1],r[1][1],b),psi(r[0][2],r[1][2],b)], 
         [chip(r[0][0],r[1][0],b),chip(r[0][1],r[1][1],b),chip(r[0][2],r[1][2],b)],
         [chim(r[0][0],r[1][0],b),chim(r[0][1],r[1][1],b),chim(r[0][2],r[1][2],b)]]
     B = [[psi(r[0][3],r[1][3],b),psi(r[0][4],r[1][4],b)], 
         [chip(r[0][3],r[1][3],b),chip(r[0][4],r[1][4],b)]]
     return np.linalg.det(A)*np.linalg.det(B)*Jastrow(rij,aligned)

def Jastrowpsi6(r,b):
     part=6
     xij=np.zeros(int(part*(part-1)/2))
     yij=np.zeros(int(part*(part-1)/2))
     k=0
     aligned = np.zeros(int(part * (part - 1) / 2), dtype=bool)
     for i in range(len(r[0])):
          for j in range(i):
               xij[k]=r[0][i]-r[0][j]
               yij[k]=r[1][i]-r[1][j]
               if i < 3 and j <3 or i >= 3 and j>=3:
                   aligned[k] = True
               else:
                   aligned[k] = False
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
     return np.linalg.det(A)*np.linalg.det(B)*Jastrow(rij,aligned)

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

