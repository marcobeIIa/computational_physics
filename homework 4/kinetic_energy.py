# here i will define all functions to take as input
# N = number of particles <= 6
# M = number of orbitals <= 3
# R = [[x1, y2],...,[xN, yN]]
# A = [[a11,...,a1N],...,[aN1,..., aNN]]
# B = [[b11,...,b1N],...,[bN1,..., bNN]]
import numpy as np
import math

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

def jastrow(N,R,N_up,b_par,b_orth):
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
    return jastrow_factor

#def jastrow_laplacianb(N,N_up,R,b_par,b_orth):
    #'''
    #b_par, b_orth : jastrow parameters for parallel / antiparallel spins
    #returns: laplacian of jastrow factor
    #'''
    #out = 0
    #for i in range(N):
        #vect_term = np.zeros(2)
        #for j in range(i,N):
            #if j != i:
                #rij = np.linalg.norm(R[i] - R[j])
                #spin_alignment = 1 if (i < N_up and j < N_up) or (i >= N_up and j >= N_up) else 0
                #aij = a_ij(spin_alignment)
                #bij = b_ij(spin_alignment, b_par, b_orth)
                #x = 1+bij*rij
                #f_ij_2der = -2*aij*bij/ x**3 + aij**2/x**4
                #f_ij_der = aij/x**2
                #vect_term[0] += f_ij_der * (R[i][0] - R[j][0]) / rij
                #vect_term[1] += f_ij_der * (R[i][1] - R[j][1]) / rij
                #out += f_ij_2der - f_ij_der**2 + f_ij_der / rij
            #else:
                #continue
        #out += vect_term[0]**2 + vect_term[1]**2
    #jassy = jastrow(N,R,N_up, b_par, b_orth)
    #out *= jassy
    #return out


def jastrow_laplacian(N, N_up,R, b_par, b_orth):
    '''
    b_par, b_orth : jastrow parameters for parallel / antiparallel spins
    returns: laplacian of jastrow factor
    '''
    R = np.asarray(R, float)          # shape (N, 2)
    N  = len(R)
    g  = np.zeros((N, 2))
    lam = np.zeros(N)

    for i in range(N):
        for j in range(i+1, N):
            r_vec = R[i] - R[j]
            r     = np.linalg.norm(r_vec)
            spin  = 1 if (i < N_up) == (j < N_up) else 0
            a     = a_ij(spin)
            b     = b_ij(spin, b_par, b_orth)
            x     = 1 + b * r

            u1 =  a / x**2                      # u′(r)
            u2 = -2*a*b / x**3                 # u″(r)

            # gradient contributions
            g_ij = u1 * r_vec / r
            g[i] +=  g_ij
            g[j] -=  g_ij                      # opposite sign for particle j

            # laplacian‑of‑log contributions (2‑D → +u′/r term)
            lam_term = u2 + u1 / r
            lam[i] += lam_term
            lam[j] += lam_term

    J = jastrow(N,R, N_up, b_par, b_orth)
    total_lap = np.sum(lam + np.sum(g**2, axis=1))
    return J * total_lap

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
        print("dis is wong")
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
    if m == 1:
        return factor * np.array([1-x**2/sigma**2,
                                 -x*y/sigma**2])
    elif m == -1:
        return factor * np.array([-y*x/sigma**2,
                                 1- y**2/sigma**2])
    else:
        raise ValueError("Invalid value for m = +-1")


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

def jastrow_f_ij(r_ij, spin_alignment, b_ij):
    '''
    r_ij             :  relative position between i, j particles
    spin_alignmnent  :  1 if i, j particles have the same spin, 0 if they have opposite spins
    '''
    aij = a_ij(spin_alignment)
    return np.exp(aij * r_ij / (1+ b_ij * r_ij) )

from functools import partial

def total_wf(N,N_up, R, sigma, b_par, b_orth, use_chi=True, return_A = True, return_A_alt = False):
    '''
    N             :  number of particles
    N_up          :  number of up-spin particles
    R             :  [[x1,y1],...,[xN,yN]] coordinates of particles
    b_par,
    b_orth        :  jastrow parameters
    sigma         :  the usual
    use_chi       :  True if you want to use the chi_m wavefunction, False if you want to use phi_nlm
    return_A      :  If True, also return the A matrix used in the determinant calculation
    return_A_alt  :  If True, also return the A matrix used in the determinant calculation, but with
                      alternative order of phi_minus and phi_plus. this is for the M=2 case, where 
                      the gradient-gradient term of the kinetic energy depends on orbital choice,
                      which is undesired and we compute it both ways to then average the result.
        
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
        elif return_A_alt:
            if N_up == 2:
                det_up,A_up = slater_det(N_up, R[:N_up], phi0, phi_minus, phi_plus, return_A = True)
                det_down,A_down = slater_det(N_down, R[N_up:], phi0, phi_plus, phi_minus, return_A = True)
            elif N_down ==2:
                det_up,A_up = slater_det(N_up, R[:N_up], phi0, phi_plus, phi_minus, return_A = True)
                det_down,A_down = slater_det(N_down, R[N_up:], phi0, phi_minus, phi_plus, return_A = True)
            else:
                print("N_up not 2, N_down not 2, why would you need return_A_alt?")
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

        psi = det_up * det_down * jastrow_factor
        if return_A or return_A_alt:
            return psi, det_up, det_down, A_up, A_down
        else:
            return psi, det_up, det_down
    else:
        raise ValueError("to be implemented.")


def slater_gradient(M,R,A_inv,i,sigma,use_chi=True):
    ''' 
    This function calculates the gradient of the Slater determinant
    A_inv - the inverse of the Slater matrix
    i - i-th partiche wrt which we're computing the gradient 
    det - the Slater determinant itself
    returns:
    - (nabla_i Det) / Det
    '''
    out = np.zeros(2)
    alpha = [[0, 0, 0], [0, 1, 1], [0, 1, -1]]  # Assuming three basis functions
    alpha_alt = [[0, 0, 0], [0, 1, -1], [0, 1, 1]]  # Assuming three basis functions
    for j in range(M):
        #print(i,j , "run\n", "A_inv =", A_inv, "\n alpha=",alpha, "\n R = ",R)
        out += A_inv[j,i] * gradient_single_particle_wf(alpha[j][2], R[i], sigma,use_chi=use_chi)
    #print("slater gradient . . . ",out)
    return out

def jastrow_grad_anal(N,N_up,R,i,b_par,b_orth):
    jastrow_grad_piece = np.zeros(2)
    for j in range(N):
        if i == j:
            continue
        else:
            rij = np.linalg.norm(R[i] - R[j])
            spin_alignment = 1 if (i < N_up and j < N_up) or (i >= N_up and j >= N_up) else 0
            aij = a_ij(spin_alignment)
            bij = b_ij(spin_alignment, b_par, b_orth)
            x = 1+bij*rij
            jastrow_prefactor = aij/ x**2 
            jastrow_grad_piece += jastrow_prefactor* (R[i]-R[j])/rij #this guy should be a vector
    return jastrow_grad_piece 

def gradient_gradient_term(N,N_up,R,
                           A_inv_up,A_inv_down,
                           b_par,b_orth,sigma,use_chi=True):
    '''
    This function calculates the gradient gradient term of jastrow, or (2)
    input:
    N          : Number of particles
    N_up       : Number of up-spin particles
    R          : Array of shape (N, 2), each row is [x, y] of particle
    A_inv_up   : Inverse of the Slater matrix for up-spin particles
    A_inv_down : Inverse of the Slater matrix for up-spin particles
    b_par      : Jastrow parameter for parallel spins
    b_orth     : Jastrow parameter for antiparallel spins
    output:
    - gradient gradient term of the full laplacian
    '''
    out = 0

    for i in range(N):
#        jastrow_grad_piece = np.zeros(2)
        #for j in range(N):
            #if i == j:
                #continue
            #else:
                #rij = np.linalg.norm(R[i] - R[j])
                #spin_alignment = 1 if (i < N_up and j < N_up) or (i >= N_up and j >= N_up) else 0
                #aij = a_ij(spin_alignment)
                #bij = b_ij(spin_alignment, b_par, b_orth)
                #sij = 1 if i < j else -1
                #x = 1+bij*rij
                #jastrow_prefactor = aij/ x**2 *sij
                #jastrow_grad_piece += jastrow_prefactor* (R[i]-R[j])/rij #this guy should be a vector
        jastrow_grad_piece = jastrow_grad_anal(N,N_up,R,i,b_par,b_orth)
        if i < N_up:
            slater_grad = slater_gradient(N_up,R[:N_up],A_inv_up,i,sigma,use_chi)
        else:
            slater_grad = slater_gradient(N-N_up,R[N_up:],A_inv_down,i-N_up,sigma,use_chi)
        out += jastrow_grad_piece[0] * slater_grad[0] + jastrow_grad_piece[1] * slater_grad[1]
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
            out += A_inv[j,i] * (-2*eigenvalue+omega**2 *r_i**2) * single_particle_wf(alpha[j][2], R[i], sigma)
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
    - b_orth    : Jastrow parameter for antiparallel spins
    - omega     : Harmonic oscillator frequency (default 1)
    - use_chi   : If True, use chi_m wavefunction, otherwise use phi_nlm
    output:
    integrand of the kinetic energy operator, i.e.
    psi lapl psi
    '''
    if N ==4:
        raise ValueError("N=4 is not implemented yet, use N=1,2,3,5,6")
    N_down = N - N_up
    psi,det_up,det_down,A_up,A_down = total_wf(N, N_up, R, sigma, b_par, b_orth, use_chi, return_A=True)
    A_up_inv = safe_invert_matrix(A_up)
    A_down_inv = safe_invert_matrix(A_down)

    jastrow_laplacian_fact = jastrow_laplacian(N, N_up, R, b_par, b_orth)
    laplacian_term_1 = det_up*det_down*jastrow_laplacian_fact

    grad_grad_term = gradient_gradient_term(N, N_up, R,
                                                A_up_inv, A_down_inv,
                                                b_par, b_orth, sigma, use_chi)

#alternative calculation with alternative oribtals for the gradient-gradient term
    if N_up == 2 or N_down ==2:
        psi_alt,det_up,det_down,A_up_alt,A_down_alt = total_wf(N, N_up, R, sigma, b_par, b_orth, use_chi, return_A=False, return_A_alt=True)
        A_up_inv_alt = safe_invert_matrix(A_up_alt)
        A_down_inv_alt = safe_invert_matrix(A_down_alt)
        grad_grad_term_2 = gradient_gradient_term(N, N_up, R,
                                                    A_up_inv_alt, A_down_inv_alt,
                                                    b_par, b_orth, sigma, use_chi)
        laplacian_term_2 = 1/2*(psi*grad_grad_term+psi_alt*grad_grad_term_2)
        print("laplacian_term_2", laplacian_term_2,
              "grad_grad_term", psi*grad_grad_term,
              "grad_grad_term_2", psi_alt*grad_grad_term_2)
    else:
        laplacian_term_2 = psi*grad_grad_term
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


def numerical_integrand(Psi,N, R, h=1e-4,return_laplacian = False):
    '''
    Numerically estimate the total Laplacian of Psi at R (N,2)-dimensional point).
    
    Parameters:
    - Psi : function R -> float, the total wavefunction
    - R   : np.array of shape (N,2), position of all particles
    - h   : finite difference step size

    Returns:
    - laplacian : float, estimate of \sum_i \nabla^2_i Psi(R)
    ''' 
    laplacian = 0.0
    for i in range(N):      # x and y
        for j in range(2):  # particle index
            dR = np.zeros_like(R)
            dR[i,j] = h
            laplacian += (Psi(R + dR) - 2 * Psi(R) + Psi(R - dR)) / h**2
    integrand = laplacian * Psi(R)
    if return_laplacian == False:
        return integrand
    else:
        return laplacian


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


# second calculation approach
def gradient_J_squared(N,N_up,R,b_par,b_orth):
    '''
    (grad J)^2, i.e. the jastrow factor
    input:
    - N         : Number of particles
    - N_up      : Number of up-spin particles
    - R         : Array of shape (N, 2), each row is [x, y] of particle
    - b_par     : Jastrow parameter for parallel spins
    - b_orth    : Jastrow parameter for antiparallel spins
    output:
    - (grad J)^2, needs a Det_up Det_down to compute the actual contribution
    '''
    log_out = 0
    for i in range(N):
        for j in range(i+1,N):
            rij = np.linalg.norm(R[i] - R[j])
            spin_alignment = 1 if (i < N_up and j < N_up) or (i >= N_up and j >= N_up) else 0
            aij = a_ij(spin_alignment)
            bij = b_ij(spin_alignment, b_par, b_orth)
            x = 1+bij*rij            
            log_jastrow_prefactor = np.abs(np.log(aij)  - 2 * np.log(x) + aij*rij / x)
            log_out +=2*log_jastrow_prefactor
    log_out += np.log(2)*N*(N-1)/2 #number of pairs
    out = np.exp(log_out)
    return out

def slater_gradient_squared(M,R,A_inv,sigma,use_chi=True):
    ''' 
    This function calculates the square gradient of the Slater determinant
    A_inv - the inverse of the Slater matrix
    i - i-th partiche wrt which we're computing the gradient 
    det - the Slater determinant itself
    returns:
    - sum_j(nabla_j Det)^2 / Det^2
    '''
    out = np.zeros(2)
    alpha = [[0, 0, 0], [0, 1, 1], [0, 1, -1]]  # Assuming three basis functions
    alpha_alt = [[0, 0, 0], [0, 1, -1], [0, 1, 1]]  # Assuming three basis functions
    for i in range(M):
        if M == 1 or M == 3:
            for j in range(M):
                #print(i,j , "run\n", "A_inv =", A_inv, "\n alpha=",alpha, "\n R = ",R)
                out += (A_inv[i][j] * gradient_single_particle_wf(alpha[j][2], R[i], sigma,use_chi=use_chi))**2
            #print("slater gradient . . . ",out)
        elif M==2: # if we only have a particle in the degenerate states, we average out the expectation value!
            out_1 = 0 
            out_2 = 0
            for j in range(M):
                #rint(i,j , "run*\n", "A_inv =", A_inv, "\n alpha=",alpha, "\n R = ",R)
                out_1 += (A_inv[i][j] * gradient_single_particle_wf(alpha[j][2], R[i], sigma,use_chi=use_chi))**2
                out_2 += (A_inv[i][j] * gradient_single_particle_wf(alpha_alt[j][2], R[i], sigma,use_chi=use_chi))**2
            out += (out_1 + out_2) / 2
    #       print("slater gradient* . . . ",out)
        else:
            out = 1.
    #       print("asdsfasdfasdfasf")
    return out

def cross_term(N,N_up,R,
               A_inv_up,A_inv_down,
               sigma,b_par,b_orth):
    '''
    cross term in (grad psi) ^2, something like grad J * grad (Det Det)
    input:
    - N         : Number of particles
    - N_up      : Number of up-spin particles
    - R         : Array of shape (N, 2), each row is [x, y] of particle
    - A_inv_up  : Inverse of the Slater matrix for up-spin particles
    - A_inv_down: Inverse of the Slater matrix for down-spin particles
    - sigma     : Sigma parameter for the harmonic oscillator wavefunction
    - b_par     : Jastrow parameter for parallel spins
    - b_orth    : Jastrow parameter for antiparallel spins
    output:
    - cross term / (psi D_up D_down) (to be reintegrated)
    '''
    out = 1
    for i in range(N):
        for j in range(i+1,N):
            rij = np.linalg.norm(R[i] - R[j])
            spin_alignment = 1 if (i < N_up and j < N_up) or (i >= N_up and j >= N_up) else 0
            aij = a_ij(spin_alignment)
            bij = b_ij(spin_alignment, b_par, b_orth)
            x = 1+bij*rij            
            jastrow_prefactor =aij/x**2 * np.exp(aij*rij / x)
            if i < N_up and j < N_up:
                spin_v_factor = slater_gradient(N_up,R[:N_up],A_inv_up,i,sigma,use_chi=True) - slater_gradient(N_up,R[:N_up],A_inv_up,j,sigma,use_chi=True)
            if i < N_up and j > N_up:
                spin_v_factor = slater_gradient(N_up,R[:N_up],A_inv_up,i,sigma,use_chi=True) - slater_gradient(N-N_up,R[N_up:],A_inv_down,j,sigma,use_chi=True)
            else:
                spin_v_factor = slater_gradient(N-N_up,R[N_up:],A_inv_down,i,sigma,use_chi=True) - slater_gradient(N-N_up,R[N_up:],A_inv_down,j,sigma,use_chi=True)
            scalar_product = 0
            for l in range(2):
                scalar_product+= spin_v_factor[l] * (R[i][l] - R[j][l])/rij
            out*= scalar_product * jastrow_prefactor
    out *= 2
    return out

def gradient_squared_full_term(N,N_up,R,sigma,b_par,b_orth,use_chi=True):
    '''
    put together all the terms in the gradient-gradient bit of the second approach to kinetic energy evaluation
    input:
    - N         : Number of particles
    - N_up      : Number of up-spin particles
    - R         : Array of shape (N, 2), each row is [x, y] of particle
    - sigma     : Sigma parameter for the harmonic oscillator wavefunction
    - b_par     : Jastrow parameter for parallel spins
    - b_orth    : Jastrow parameter for antiparallel spins
    - use_chi   : If True, use chi_m wavefunction, otherwise use phi_nlm
    output:
    - gradient gradient term of the full laplacian
    '''
    psi,det_up,det_down,A_up,A_down = total_wf(N, N_up, R, sigma, b_par, b_orth, use_chi, return_A=True)

    term_1 = gradient_J_squared(N,N_up,R,b_par,b_orth) * det_up * det_down
    sl_gr_sr = slater_gradient_squared(N_up, R[:N_up], A_up, sigma, use_chi) + slater_gradient_squared(N-N_up, R[N_up:], A_down, sigma, use_chi)
    term_2 = sl_gr_sr * psi **2
    term_3 = cross_term(N,N_up,R,A_up,A_down,sigma,b_par,b_orth) * psi * det_up * det_down
    return term_1 + term_2 + term_3

def kinetic_energy_integrand_2(N,N_up,R,sigma,b_par,b_orth,omega=1,use_chi=True): 
    '''
    second way of computing the kinetic energy, with
    1/2 * (psi lapl psi -  nabla^2 psi / psi)
    in place of psi lapl psi.
    input: same as kinetic_energy_integrand
    output: integrand of the kinetic energy operator, same as kinetic_energy_integrand
    '''
    return 1/2 * (kinetic_energy_integrand(N,N_up,R,sigma,b_par,b_orth,omega,use_chi) - gradient_squared_full_term(N,N_up,R,sigma,b_par,b_orth,use_chi))











