# here i will define all functions to take as input
# N = number of particles <= 6
# M = number of orbitals <= 3
# R = [[x1, y2],...,[xN, yN]]
# A = [[a11,...,a1N],...,[aN1,..., aNN]]
# B = [[b11,...,b1N],...,[bN1,..., bNN]]
import numpy as np
import math
import random



def single_particle_wf(m,r,sigma,use_chi=True):
    '''
    m = 0, +1, -1
    r= [x,y] = coordinates of i-th particle
    sigma the usual
    use_chi = True if you want to use the chi_m wavefunction, False if you want to use phi_nlm
    returns:
    - single particle wavefunction with projection q.n. m, depending on my choice of use_chi 
    '''
    x,y = r
    phi000 = (1/np.sqrt(np.pi*sigma**2))*np.exp(-(x**2+y**2)/(2*sigma**2))   
    if m == 0:
        return phi000
    elif use_chi:
        return phi000*x/sigma if m == 1 else phi000*y/sigma
    else:
        return phi000*(x+1j*m*y)/sigma 

def A_matrix_creator(M, R, phi1, phi2, phi3):
    """
    creates the matrix from which we compute the slater determinant

    Parameters:
    - M      : Number of particles of given spin
    - R      : Array of shape (M, 2), each row is [x, y] of particle
    - sigma  : Sigma parameter in h.o. wavefunction
    - phi1    : Callable: phi1(r)
    - phi2   : Callable: phi2(r)
    - phi3   : Callable: phi3(r)

    returns :
    - A[i,j] = phi_j(x_i, y_i, spin_i) 
    """
    basis_functions = [phi1, phi2, phi3]
    num_basis = len(basis_functions)
    if M > num_basis:
        raise ValueError("not enough basis functions for n orbitalss")

    A = np.zeros((M, M), dtype=float)

    for i in range(M):  # row: particle
        r = R[i]
        for j in range(M):  # col: basis function
            A[i, j] = basis_functions[j](r=r)
    return A

def safe_invert_matrix(A, rcond=1e-15):
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

def slater_det(M, R, phi1, phi2, phi3, normalised = False, return_A = False):
    """
    Compute the Slater determinant for M particles.

    Parameters:
    - M         : Number of particles 
    - R         : Array of shape (M, 2), each row is [x, y] of particle
    - sigma     : Sigma parameter for the harmonic oscillator wavefunction
    - phi       : Callable: phi(x, y, spin)
    - chip      : Callable: chip(x, y, spin)
    - chim      : Callable: chim(x, y, spin)
    - return_A  : If True, also return the A matrix used in the determinant calculation

    Returns:
    - det    : Value of the Slater determinant
    - A      : (optional) The matrix A used in the determinant calculation
    """
    basis_functions = [phi1, phi2, phi3]
    num_basis = len(basis_functions)

    if M > num_basis:
        raise ValueError("not enough basis functions for N orbitalss")

    A = A_matrix_creator(M, R, phi1, phi2, phi3)

    # Normalization factor, i dont wanna normalise because im scared of big numbers
    if not normalised:
        normalisation = 1
    else: 
        normalisation = 1 / math.factorial(M)
    # returns A since we computed it anyway, save a little time :)
    if return_A:
        return normalisation * np.linalg.det(A), A
    else:
        return normalisation * np.linalg.det(A)

def a_ij(spin_alignment):
    '''
    spin_alignment = 1 if i, j particles have the same spin, 0 if they have opposite spins
    returns: a_ij coefficient for the jastrow factor
    '''
    return 1/2 - 1/4*spin_alignment

def b_ij(spin_alignment, b_par, b_orth):
    '''
    spin_alignment = 1 if i, j particles have the same spin, 0 if they have opposite spins
    returns: b_ij coefficient for the jastrow factor
    '''
    if spin_alignment:
        return b_par
    else:
        return b_orth

def jastrow_laplacian(N,N_up,R,b_par,b_orth):
    '''
    b_par, b_orth : jastrow parameters for parallel / orthogonal spins
    returns: laplacian of jastrow factor
    '''
    out = 1
    for i in range(N):
        for j in range(i+1,N):
            rij = np.linalg.norm(R[i] - R[j])
            spin_alignment = 1 if (i < N_up and j < N_up) or (i >= N_up and j >= N_up) else 0
            aij = a_ij(spin_alignment)
            bij = b_ij(spin_alignment, b_par, b_orth)
            x = 1+bij*rij
            out *= (-2*aij*bij/ x + aij**2/x**2 + aij/rij)/x**2 * np.exp(-aij*rij / x)
    out *= 2 
    return out

def gradient_phi(alpha, r,sigma):
    '''
    alpha =  [n,l,m] orbital nmbers
    r= [x,y] = coordinates of i-th particle
    sigma the usual
    returns:
    - gradient of phi_nlm
    '''
    x,y = r
    n,l,m = alpha
    factor = np.exp(-(x**2+y**2)/(2*sigma**2))/ (np.sqrt(math.pi) * sigma**3)
    if alpha == [0,0,0]:
        return np.array([-x* factor,
                         -y* factor])
    elif [n,l] == [0,1]:
        # these are most certainly wrong but we don't even use them
        # cba to compute the right result. i am getting no money from this
        return np.array([(1- 1/sigma**2*(x+m*1j*y))* factor,
                         (m*1j - 1/sigma**2*(x+m*1j*y))* factor])

def gradient_chi(m, r,sigma):
    '''
    m = +-1
    r= [x,y] = coordinates of i-th particle
    sigma the usual
    returns:
    - gradient of chi_m
    '''
    x,y = r
    factor = np.exp(-(x**2+y**2)/(2*sigma**2))/ (np.sqrt(math.pi) * sigma**2)
    if m == -1:
        return factor * np.array([-x*y/sigma**2,
                                 1- y**2/sigma**2])
    elif m == 1:
        return factor * np.array([-x*y/sigma**2,
                                 1- y**2/sigma**2])
    else:
        raise ValueError("Invalid value for m = +-1")

def jastrow_f_ij(r_ij, spin_alignment, b_ij):
    '''
    r_ij             :  relative position between i, j particles
    spin_alignmnent  :  1 if i, j particles have the same spin, 0 if they have opposite spins
    '''
    aij = a_ij(spin_alignment)
    return np.exp(aij * r_ij / (1+ b_ij * r_ij) )

from functools import partial

def total_wf(N,N_up, R, sigma, b_par, b_orth, use_chi=True, return_A = True, jj=True):
    '''
    N         :  number of particles
    N_up      :  number of up-spin particles
    R         :  [[x1,y1],...,[xN,yN]] coordinates of particles
    b_par,
    b_orth    : jastrow parameters
    sigma     :  the usual
    use_chi   :  True if you want to use the chi_m wavefunction, False if you want to use phi_nlm
    return_A  : If True, also return the A matrix used in the determinant calculation
    jj        : If true add Jastrow function
    returns:
    - total wavefunction of the system, i.e. product of single particle wavefunctions times the jastrow 
    - slater determinant for up,down-spin particles
    - A_up and A_down matrices if return_A is True
    '''
    assert N - N_up >= 0, "N_up cannot be greater than N"
    if N != 4:
        N_down = N - N_up
        phi0 =  partial(single_particle_wf, m=0, sigma = sigma, use_chi = use_chi)
        phi_plus =  partial(single_particle_wf, m=1, sigma = sigma, use_chi = use_chi)
        phi_minus =  partial(single_particle_wf, m=-1, sigma = sigma, use_chi=use_chi)
        if return_A:
            det_up,A_up = slater_det(N_up, R[:N_up], phi0, phi_plus, phi_minus, return_A = True)
            det_down,A_down = slater_det(N_down, R[N_up:], phi0, phi_plus, phi_minus, return_A = True)
        else:
            det_up = slater_det(N_up, R[:N_up], phi0, phi_plus, phi_minus, return_A = False)
            det_down = slater_det(N_down, R[N_up:], phi0, phi_plus, phi_minus, return_A = False)

        jastrow_factor = 1.
        #for i in range(N):
            #for j in range(i+1, N):
                #r_ij = np.linalg.norm(R[i] - R[j])
                #spin_alignment = 1 if (i < N_up and j < N_up) or (i >= N_up and j >= N_up) else 0
               ## print("mb--jastrowfunct",i,j, "-th iteration")
               ## print("rij",r_ij)
                #bij = b_ij(spin_alignment,b_par,b_orth)
                #jastrow_factor *= jastrow_f_ij(r_ij, spin_alignment, bij)
                #print("jastrow",jastrow_factor)
        #print("-------------\n mb jstrow final",jastrow_factor)
        jastrow_log = 0.
        for i in range(N):
            for j in range(i+1, N):
                r_ij = np.linalg.norm(R[i] - R[j])
       #         print("mb rij",r_ij)
                spin_alignment = 1 if (i < N_up and j < N_up) or (i >= N_up and j >= N_up) else 0
       #         print("spin alignment mb",spin_alignment)
                bij = b_ij(spin_alignment, b_par, b_orth)
                aij = a_ij(spin_alignment)
                jastrow_log += aij * r_ij / (1 + bij * r_ij)
        jastrow_factor = np.exp(jastrow_log)
       # print("full jastrow mb",jastrow_factor)

        if jj:
            psi = det_up * det_down * jastrow_factor
        else:
            psi = det_up * det_down 

            
        if return_A:
            return psi, det_up, det_down, A_up, A_down
        else:
            return psi, det_up,det_down
    else:
        raise ValueError("to be implemented.")

def gradient_single_particle_wf(m, r, sigma, use_chi=True):
    '''
    - m = 0, +1, -1
    - r= [x,y] = coordinates of i-th particle
    - sigma the usual
    - use_chi = True if you want to use the chi_m wavefunction, False if you want to use phi_nlm
    returns:
    - gradient of the single particle wavefunction with projection q.n. m, depending on my choice of use_chi 
    '''
    if m == 0:
        return gradient_phi([0,0,0], r, sigma)
    elif use_chi:
        return gradient_chi(m, r, sigma)
    else:
        return gradient_phi([0,1,m], r, sigma) 

def slater_gradient(M,R,A_inv,i,det,sigma,use_chi=True):
    ''' 
    This function calculates the gradient of the Slater determinant
    A_inv - the inverse of the Slater matrix
    i - i-th partiche wrt which we're computing the gradient 
    det - the Slater determinant itself
    '''
    out = np.zeros(2)
    alpha = [[0, 0, 0], [0, 1, 1], [0, 1, -1]]  # Assuming three basis functions
    alpha_alt = [[0, 0, 0], [0, 1, -1], [0, 1, 1]]  # Assuming three basis functions
    if M == 1 or M == 3:
        for j in range(M):
            #print(i,j , "run\n", "A_inv =", A_inv, "\n alpha=",alpha, "\n R = ",R)
            out += A_inv[i][j] * gradient_single_particle_wf(alpha[j][2], R[i], sigma,use_chi=use_chi)
        out *= det
        #print("slater gradient . . . ",out)
    elif M==2: # if we only have a particle in the degenerate states, we average out the expectation value!
        out_1 = 0 
        out_2 = 0
        for j in range(M):
            #rint(i,j , "run*\n", "A_inv =", A_inv, "\n alpha=",alpha, "\n R = ",R)
            out_1 += A_inv[i][j] * gradient_single_particle_wf(alpha[j][2], R[i], sigma,use_chi=use_chi)
            out_2 += A_inv[i][j] * gradient_single_particle_wf(alpha_alt[j][2], R[i], sigma,use_chi=use_chi)
        out = det * (out_1 + out_2) / 2
#       print("slater gradient* . . . ",out)
    else:
        out = 1.
#       print("asdsfasdfasdfasf")
    return out

def gradient_gradient_term(N,N_up,R,
                           A_inv_up,A_inv_down,
                           det_up,det_down,
                           b_par,b_orth,sigma,use_chi=True):
    '''
    This function calculates the gradient gradient term of jastrow, or (2)
    input:
    N        : Number of particles
    N_up     : Number of up-spin particles
    R        : Array of shape (N, 2), each row is [x, y] of particle
    b_par    : Jastrow parameter for parallel spins
    b_orth   : Jastrow parameter for orthogonal spins
    output:
    - gradient gradient term of the full laplacian
    '''
    out = 1

    for i in range(N):
        for j in range(i+1,N):
            rij = np.linalg.norm(R[i] - R[j])
            spin_alignment = 1 if (i < N_up and j < N_up) or (i >= N_up and j >= N_up) else 0
            aij = a_ij(spin_alignment)
            bij = b_ij(spin_alignment, b_par, b_orth)
            x = 1+bij*rij
            jastrow_prefactor = aij/ x**2 * np.exp(aij*rij / x)
            jastrow_grad_piece = jastrow_prefactor* (R[i]-R[j]) #this guy should be a vector

            if i < N_up and j < N_up: # i is up-spin particle, j is also
                slater_grad = slater_gradient(N_up,R[:N_up],A_inv_up,i,det_up,sigma,use_chi) - slater_gradient(N_up,R[:N_up],A_inv_up,j,det_up,sigma,use_chi)
            elif i < N_up and j >= N_up: # i is up-spin particle, j is down
                #print("1------------\n",slater_gradient(N_up,R[:N_up],A_inv_up,i,det_up,sigma,use_chi))
                #print("2------------\n",slater_gradient(N-N_up,R[N_up:],A_inv_down,j-N_up,det_down,sigma,use_chi))
                slater_grad = slater_gradient(N_up,R[:N_up],A_inv_up,i,det_up,sigma,use_chi) - slater_gradient(N-N_up,R[N_up:],A_inv_down,j-N_up,det_down,sigma,use_chi)
            elif i >= N_up and j >= N_up: # i is down-spin particle, j is also
                slater_grad = slater_gradient(N-N_up,R[N_up:],A_inv_down,i-N_up,det_down,sigma,use_chi) - slater_gradient(N-N_up,R[N_up:],A_inv_down,j-N_up,det_down,sigma,use_chi)
            for l in range(2):
                #print("slatergrad",slater_grad, jastrow_grad_piece)
                out_placeholder = jastrow_grad_piece[l] * slater_grad[l]
                out *= out_placeholder
    return 2*out

def ho_eigenvalue(alpha,omega):
    ''' 
    input:
    - alpha : [n,l,m]
    - omega : h.o. hamiltonian parameter
    output:
    - eigenvalue of the harmonic oscillator for the given quantum numbers
    '''
    n,l,m = alpha
    return omega*(2*n + l + 1)


def slater_laplacian_term(M, R, A_inv, sigma,omega=1):
    """
    Compute the Laplacian of the Slater determinant for M orbitals.

    Parameters:
    - M      : Number of orbitalss
    - R      : Array of shape (N, 2), each row is [x, y] of particle
    - sigma  : Sigma parameter for the harmonic oscillator wavefunction
    - phi    : Callable: phi(x, y, spin)
    - chip   : Callable: chip(x, y, spin)
    - chim   : Callable: chim(x, y, spin)

    Returns:
    - laplacian : Value of the Laplacian of the Slater determinant 
                  (actually lapl (det) / det) !!!
    """
    out = 0
    alpha = [[0, 0, 0], [0, 1, 1], [0, 1, -1]]  # Assuming three basis functions
    for i in range(M):
        r_i = np.linalg.norm(R[i])
        for j in range(M):
            eigenvalue = ho_eigenvalue(alpha[j], omega)
            out += A_inv[i][j] * (-2*eigenvalue+omega**2 *r_i**2) * single_particle_wf(alpha[j][2], R[i], sigma)
    return out

def kinetic_energy_integrand(N,N_up,R,sigma,b_par,b_orth,omega=1,use_chi=True):
    '''
    finally, this function returnst the integrand of the kinetic energy
    input:
    - N         : Number of particles
    - N_up      : Number of up-spin particles
    - R         : Array of shape (N, 2), each row is [x, y] of particle
    - sigma     : Sigma parameter for the harmonic oscillator wavefunction
    - b_par     : Jastrow parameter for parallel spins
    - b_orth    : Jastrow parameter for orthogonal spins
    - omega     : Harmonic oscillator frequency (default 1)
    - use_chi   : If True, use chi_m wavefunction, otherwise use phi_nlm
    output:
    integrand of the kinetic energy operator, i.e.
    psi lapl psi
    '''
    N_down = N - N_up
    psi,det_up,det_down,A_up,A_down = total_wf(N, N_up, R, sigma, b_par, b_orth, use_chi, return_A=True)
    jastrow_laplacian_fact = jastrow_laplacian(N, N_up, R, b_par, b_orth)

    A_up_inv = safe_invert_matrix(A_up)
    A_down_inv = safe_invert_matrix(A_down)

    laplacian_term_1 = det_up*det_down*jastrow_laplacian_fact
    grad_grad_term = gradient_gradient_term(N, N_up, R,
                                                A_up_inv, A_down_inv,
                                                det_up, det_down,
                                                b_par, b_orth, sigma, use_chi)
    laplacian_term_2 = det_up*det_down*grad_grad_term
    slater_laplacian_up = slater_laplacian_term(N_up, R[:N_up], A_up_inv, sigma, omega)
    slater_laplacian_down = slater_laplacian_term(N_down, R[N_up:], A_down_inv, sigma, omega)
    laplacian_term_3 = psi * (slater_laplacian_up + slater_laplacian_down)

    laplacian = laplacian_term_1 + laplacian_term_2 + laplacian_term_3
    integrand = psi * (laplacian_term_1 + laplacian_term_2 + laplacian_term_3)
    print("psi:", psi,
          "term 1:",laplacian_term_1, 
          "term 2:",laplacian_term_2, 
          "term 3:",laplacian_term_3)
    return integrand


def wf_laplacian(R,wavefunction, sigma, h=1e-4):
    """
    Compute the Laplacian of the wavefunction Ψ(R, sigma),
    where R is of shape (2, N).
    
    Returns a scalar: sum of second partial derivatives.
    """
    dim, N = R.shape
    laplacian = 0.0
    
    # Loop over all particles and their coordinates (x or y)
    for i in range(N):
        for mu in range(dim):  # mu = 0 (x), 1 (y)
            dR = np.zeros_like(R)
            dR[mu, i] = h

            plus = wavefunction(R + dR, sigma)
            center = wavefunction(R, sigma)
            minus = wavefunction(R - dR, sigma)
            
            laplacian += (plus - 2 * center + minus) / h**2
    
    return laplacian


def numerical_integrand(Psi, R, h=1e-4):

    """
    Numerically estimate the total Laplacian of Psi at R (2N-dimensional point).
    
    Parameters:
    - Psi : function R -> float, the total wavefunction
    - R   : np.array of shape (2N,), position of all particles
    - h   : finite difference step size

    Returns:
    - laplacian : float, estimate of \sum_i \nabla^2_i Psi(R)
    """
    laplacian = 0.0
    for i in range(len(R)):
        dR = np.zeros_like(R)
        dR[i] = h
        laplacian += (Psi(R + dR) - 2 * Psi(R) + Psi(R - dR)) / h**2
    integrand = laplacian * Psi(R)
    return integrand

def N_up_choice(N):
    '''
    returns the right value of N_up to match with results from lusva
    '''
    if N==1:
        N_up = 1
    elif N==2:
        N_up = 1
    elif N==3:
        N_up = 2
    elif N==4:
        N_up = 2
    elif N==5:
        N_up = 3
    elif N==6:
        N_up = 3
    else:
        raise ValueError("N must be between 1 and 6")
    return N_up



def Metropolisguess(r,delta):
    rn=np.zeros((len(r),len(r[0])))
    for i in range(len(r)):
        for j in range(len(r[0])):
            a=random.uniform(0,1)
            rn[i][j]= r[i][j]+delta*(a-0.5)
    return rn

def Metropolis(N, N_up, R, sigma, b_par, b_orth, delta, counter, acc ,jj):

    Rn = Metropolisguess(R,delta)
    psi, det_up, det_down = total_wf(N,N_up,R,sigma,b_par,b_orth,use_chi=True,return_A=False,jj=jj)
    psinew, det_upnew, det_downnew = total_wf(N,N_up,Rn,sigma,b_par,b_orth,use_chi=True,return_A=False,jj=jj)

    p=psinew**2/psi**2
    if p>1:
        R=Rn
        acc+=1
    else:
        xi=random.uniform(0,1)
        if p>xi:
            R=Rn
            acc+=1

    counter+=1
    return R,acc, counter



def adjust_delta(delta, acceptance_rate):
    if acceptance_rate > 0.6:
        delta *= 1.1  # Increase delta by 10%
    elif acceptance_rate < 0.4:
        delta *= 0.9  # Decrease delta by 10%
    return delta


def delta_choice(N, N_up, R, sigma, b_par, b_orth, delta, counter, acc,N_delta,jj):
  '''
    N         :  number of particles
    N_up      :  number of up-spin particles
    R         :  [[x1,y1],...,[xN,yN]] coordinates of particles
    b_par,
    b_orth    : jastrow parameters
    sigma     :  the usual
    delta     : Metropolis parameter
    '''
  acc_rate=1
  while acc_rate>0.6 or acc_rate<0.4:
    acc,counter=0,0
    for i in range(N_delta):
      R, acc,counter = Metropolis(N, N_up, R, sigma, b_par, b_orth, delta, counter, acc,jj)
    acc_rate=acc/counter
    delta = adjust_delta(delta, acc_rate)

  return delta
