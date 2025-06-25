# here i will define all functions to take as input
# N = number of particles <= 6
# M = number of orbitals <= 3
# R = [[x1, y2],...,[xN, yN]]
# A = [[a11,...,a1N],...,[aN1,..., aNN]]
# B = [[b11,...,b1N],...,[bN1,..., bNN]]
import numpy as np
import math
def A_matrix_creator(M, R, sigma, phi, chip, chim):
    """
    creates the matrix from which we compute the slater determinant

    Parameters:
    - M      : Number of particles of given spin
    - R      : Array of shape (M, 2), each row is [x, y] of particle
    - sigma  : Sigma parameter in h.o. wavefunction
    - phi    : Callable: phi(x, y, spin)
    - chip   : Callable: chip(x, y, spin)
    - chim   : Callable: chim(x, y, spin)

    returns :
    - A[i,j] = phi_j(x_i, y_i, spin_i) 
    """
    basis_functions = [phi, chip, chim]
    num_basis = len(basis_functions)
    if M > num_basis:
        raise ValueError("not enough basis functions for n orbitalss")

    A = np.zeros((M, M), dtype=float)

    for i in range(M):  # row: particle
        x, y = R[i]
        for j in range(M):  # col: basis function
            A[i, j] = basis_functions[j](x, y, sigma)

def slater_det(M, R, sigma, phi, chip, chim, normalised = False):
    """
    Compute the Slater determinant for M particles.

    Parameters:
    - M      : Number of particles 
    - R      : Array of shape (M, 2), each row is [x, y] of particle
    - sigma  : Sigma parameter for the harmonic oscillator wavefunction
    - phi    : Callable: phi(x, y, spin)
    - chip   : Callable: chip(x, y, spin)
    - chim   : Callable: chim(x, y, spin)

    Returns:
    - det    : Value of the Slater determinant
    """
    basis_functions = [phi, chip, chim]
    num_basis = len(basis_functions)

    if M > num_basis:
        raise ValueError("not enough basis functions for N orbitalss")

    A = A_matrix_creator(M, R, sigma, phi, chip, chim)

    # Normalization factor, i dont wanna normalise because im scared of big numbers
    if not normalised:
        normalisation = 1
    else: 
        normalisation = 1 / math.factorial(M)
    return normalisation * np.linalg.det(A)


def jastrow_laplacian(N,R,A,B):
    '''
    returns: laplacian of jastrow factor
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
    factor = np.exp((x**2+y**2)/(2*alpha[0]**2))/ sigma**2
    if alpha == [0,0,0]:
        return np.array(x* factor,
                         y* factor)
    elif [n,l] == [0,1]:
        return np.array((x+sigma)* factor,
                         (y+1j*m*sigma)* factor)

def gradient_chi(m, r,sigma):
    '''
    m = +-1
    r= [x,y] = coordinates of i-th particle
    sigma the usual
    returns:
    - gradient of chi_m
    '''
    if m == 1:
        return (gradient_phi([0,1,1], r, sigma) + gradient_phi([0,-1,1], r, sigma))/2
    elif m ==-1:
        return (gradient_phi([0,1,1], r, sigma) - gradient_phi([0,-1,1], r, sigma))/(2j)
    else:
        raise ValueError("Invalid value for m = +-1")

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
    phi000 = (1/np.sqrt(np.pi*sigma**2))*math.exp((x**2+y**2)/(2*sigma**2))   
    if m == 0:
        return phi000
    elif use_chi:
        return phi000*x/sigma if m == 1 else phi000*y/sigma
    else:
        return phi000*(x+1j*m*y)/sigma 

def jastrow_f_ij(r_ij, spin_alignment, b_ij):
    '''
    r_ij             :  relative position between i, j particles
    spin_alignmnent  :  1 if i, j particles have the same spin, 0 if they have opposite spins
    '''
    a_ij = 1/2 - 1/4*spin_alignment
    return np.exp(a_ij * r_ij / (1+ b_ij * r_ij) )

def total_wf(N,N_up, R, sigma, b_par, b_orth, use_chi=True, paramagnetic = False):
    '''
    N        :  number of particles
    N_up     :  number of up-spin particles
    R        :  [[x1,y1],...,[xN,yN]] coordinates of particles
    b_par,
    b_orth   : jastrow parameters
    sigma    :  the usual
    use_chi  :  True if you want to use the chi_m wavefunction, False if you want to use phi_nlm
    returns:
    - total wavefunction of the system, i.e. product of single particle wavefunctions times the jastrow 
    '''
    assert N - N_up >= 0, "N_up cannot be greater than N"
    if N != 4:
        N_down = N - N_up
        det_up = slater_det(N_up, R[:N_up], sigma, single_particle_wf, gradient_chi, gradient_chi)
        det_down = slater_det(N_down, R[N_up:], sigma, single_particle_wf, gradient_chi, gradient_chi)
        jastrow_factor = 1.
        for i in range(N):
            for j in range(i+1, N):
                r_ij = np.linalg.norm(R[i] - R[j])
                spin_alignment = 1 if (i < N_up and j < N_up) or (i >= N_up and j >= N_up) else 0
                b_ij = b_par * spin_alignment + b_orth * (1 - spin_alignment)
                jastrow_factor *= jastrow_f_ij(r_ij, spin_alignment, b_ij)
        psi = det_up * det_down * jastrow_factor
        return psi, det_up,det_down
    else:
        raise ValueError("to be implemented.")



def gradient_single_particle_wf(m, r, sigma, use_chi=True):
    '''
    m = 0, +1, -1
    r= [x,y] = coordinates of i-th particle
    sigma the usual
    use_chi = True if you want to use the chi_m wavefunction, False if you want to use phi_nlm
    returns:
    - gradient of the single particle wavefunction with projection q.n. m, depending on my choice of use_chi 
    '''
    if m == 0:
        return gradient_phi([0,0,0], r, sigma)
    elif use_chi:
        return gradient_chi(m, r, sigma)
    else:
        return gradient_phi([0,1,m], r, sigma) 



def slater_gradient(M,R,A_inv,i,det,sigma):
    ''' 
    This function calculates the gradient of the Slater determinant
    A_inv - the inverse of the Slater matrix
    i - i-th partiche wrt which we're computing the gradient 
    det - the Slater determinant itself
    '''
    out = np.zeros((1, 2))
    alpha = [[0, 0, 0], [0, 1, 1], [0, 1, -1]]  # Assuming three basis functions
    for j in range(M):
        out += A_inv[i][j] * gradient_single_particle_wf(alpha[j][2], R[i], sigma)
    out *= det
    return out


def gradient_gradient_term(N,R,A,B):
    '''
    This function calculates the gradient gradient term of jastrow, or (2)
    '''
    out = 0
    #for i in range(N):
        #for j in range(i+1,N):
            #rij = np.linalg.norm(R[i] - R[j])
            #aij = A[i][j]
            #bij = B[i][j]
            #x = 1+bij*rij
            #jastrow_prefactor = aij/ x**2 * np.exp(-aij*rij / x)
            #jastrow_piece = jastrow_prefactor* (R[i]-R[j]) #this guy should be a vector
    return out

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


def slater_laplacian_term(M, R, A_inv, det, sigma,omega=1):
    """
    Compute the Laplacian of the Slater determinant for N orbitals.

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
    out = np.zeros((1, 2))
    alpha = [[0, 0, 0], [0, 1, 1], [0, 1, -1]]  # Assuming three basis functions
    for i in range(M):
        r_i = np.linalg.norm(R[i])
        for j in range(M):
            eigenvalue = ho_eigenvalue(alpha[j], omega)
            out += A_inv[i][j] * (-2*eigenvalue+omega**2 *r_i**2) * single_particle_wf(alpha[j][2], R[i], sigma)
    return out

def kinetic_energy_integrand(N,N_up,R,sigma,b_par,b_orth,use_chi=True,paramagnetic=False):
    psi,det_up,det_down = total_wf(N, N_up, R, sigma, b_par, b_orth, use_chi, paramagnetic)
    grad_term_1 = 
    N_down = N - N_up