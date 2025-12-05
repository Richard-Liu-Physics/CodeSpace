'''
Created by Richard Liu, all rights reserved.
Date: Apr 15, 2025
'''
import numpy as np
import scipy.sparse as sp
import Library_RL.Pauli as Pauli

def DenMatrix_to_Pauli_based_sparse(N: int, rho_sp: sp.csr_matrix)->np.ndarray:
    '''
    Transfores a density matrix rho into a Pauli basis representation.
    Args:
        N: The number of qubits.
        rho_sp: The density matrix in sparse format.
    Returns:
        A numpy array representing the density matrix in the Pauli basis.
    '''
    # Check if the input density matrix is a trace-1 matrix
    if not rho_sp.diagonal().sum() - 1 < 1e-10:
        raise ValueError("The input density matrix must be trace-1.")
    # Initialize the Pauli basis representation
    rho_Pauli = np.zeros(4**N)
    # Create the Pauli basis matrices
    Pauli_matrices = ['I', 'X', 'Y', 'Z']
    # Iterate over all combinations of Pauli matrices
    for i in range(4**N):
        # Convert the index to a binary representation
        binary_index = np.array([int(x) for x in np.base_repr(i, base=4).zfill(N)])
        # Compute the corresponding Pauli string
        pauli_string = ' '.join([Pauli_matrices[binary_index[j]] for j in range(N)])
        # Compute the inner product with the density matrix
        rho_Pauli[i] = (rho_sp @ Pauli.Pauli_string_sparse(pauli_string)).trace()/2**N
    return rho_Pauli

def DenMatrix_to_Pauli_based(N: int, rho: np.ndarray)->np.ndarray:
    '''
    Transfores a density matrix rho into a Pauli basis representation.
    Args:
        N: The number of qubits.
        rho: The density matrix to be transformed.
    Returns:
        A numpy array representing the density matrix in the Pauli basis.
    '''
    rho_sp = sp.csr_matrix(rho)
    return DenMatrix_to_Pauli_based_sparse(N, rho_sp)

def Pauli_based_to_DenMatrix(N: int, rho_Pauli: np.ndarray)->np.ndarray:
    '''
    Transforms a density matrix in the Pauli basis back to the original density matrix.
    Args:
        N: The number of qubits.
        rho_Pauli: The density matrix in the Pauli basis.
    Returns:
        A numpy array representing the original density matrix.
    '''
    # Initialize the density matrix
    rho = np.zeros((2**N, 2**N), dtype=complex)
    # Create the Pauli basis matrices
    Pauli_matrices = ['I', 'X', 'Y', 'Z']
    # Iterate over all combinations of Pauli matrices
    for i in range(4**N):
        # Convert the index to a binary representation
        binary_index = np.array([int(x) for x in np.base_repr(i, base=4).zfill(N)])
        # Compute the corresponding Pauli string
        pauli_string = ' '.join([Pauli_matrices[binary_index[j]] for j in range(N)])
        # Compute the outer product with the density matrix
        rho += rho_Pauli[i] * Pauli.Pauli_string(pauli_string)
    return rho

def SRE(a:int, n: int, rho: np.ndarray,)->float:
    '''
    Computes the SRE of a density matrix rho.
    Args:
        a: The renyi index.
        n: The number of qubits in the system.
        rho: The density matrix.
    Returns:
        The SRE of the density matrix.
    '''
    # Compute the Pauli basis representation
    rho_Pauli = DenMatrix_to_Pauli_based(n, rho)
    tem = np.array([(x * (2**n))**(2 * a) for x in rho_Pauli])
    # Compute the SRE
    Stabilizer_purity = sum(tem)/2**n
    SRE = 1/(1-a)*np.log2(Stabilizer_purity)
    return SRE

def SRE_Pauli(a:int, n: int, rho_Pauli: np.ndarray,)->float:
    '''
    Computes the SRE of a density matrix in the Pauli basis.
    Args:
        a: The number of qubits.
        n: The number of qubits in the system.
        rho_Pauli: The density matrix in the Pauli basis.
    Returns:
        The SRE of the density matrix.
    '''
    # Compute the SRE
    tem = np.array([(x * (2**n))**(2 * a) for x in rho_Pauli])
    Stabilizer_purity = sum(tem)/2**n
    SRE = 1/(1-a)*np.log2(Stabilizer_purity)
    return SRE

def unitary_to_Pauli_based_sparse(N: int, U: sp.csr_matrix)->sp.csr_matrix:
    '''
    Transforms an evolution by a unitary matrix U into a Pauli basis representation.
    Args:
        N: The number of qubits.
        U: The unitary matrix to be transformed.
    Returns:
        A sparse matrix representing the unitary matrix in the Pauli basis.
    '''
    Uconjtrans = U.conjugate().transpose()
    # Initialize the Pauli basis representation
    U_Pauli = np.zeros((4**N, 4**N), dtype=complex)
    # Create the Pauli basis matrices
    Pauli_matrices = ['I', 'X', 'Y', 'Z']
    # Iterate over all combinations of Pauli matrices
    for i in range(4**N):
        for j in range(4**N):
            # Convert the indices to binary representations
            binary_index_i = np.array([int(x) for x in np.base_repr(i, base=4).zfill(N)])
            binary_index_j = np.array([int(x) for x in np.base_repr(j, base=4).zfill(N)])
            # Compute the corresponding Pauli strings
            pauli_string_i = ' '.join([Pauli_matrices[binary_index_i[k]] for k in range(N)])
            pauli_string_j = ' '.join([Pauli_matrices[binary_index_j[k]] for k in range(N)])
            
            temi = Pauli.Pauli_string_sparse(pauli_string_i)
            temj = Pauli.Pauli_string_sparse(pauli_string_j)
            # Compute the inner product with the unitary matrix
            U_Pauli[i, j] = (temi @ U @ temj @ Uconjtrans).trace()/2**N
    return sp.csr_matrix(U_Pauli)
