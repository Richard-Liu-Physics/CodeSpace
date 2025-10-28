'''
Created by Richard Liu, all rights reserved.
Date: Oct 14, 2025
'''
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix,kron,eye
def Pauli(i: str):
    '''
    Define the Pauli matrix. i should be 'X' or 'Y' or 'Z' or 'I'
    '''
    if i == 'I':
        return np.eye(2)
    elif i == 'X':
        return np.array([[0,1], [1,0]])
    elif i == 'Y':
        return np.array([[0, -1j], [1j, 0]])
    elif i == 'Z':
        return np.array([[1, 0], [0, -1]])

def Pauli_string(i: str) -> np.ndarray:
    'Define a Pauli string according to string i which should be as "X Y X I Z"'
    list_tem = i.split()
    tem = np.eye(1)
    for x in list_tem:
        tem = np.kron(tem, Pauli(x))
    return tem

def Pauli_string_sparse(i: str) -> csr_matrix:
    'Define a sparse Pauli string according to string i which should be as "X Y X I Z"'
    list_tem = i.split()
    tem = csr_matrix(np.eye(1))
    for x in list_tem:
        tem = kron(tem, csr_matrix(Pauli(x)), format='csr')
    return tem

def Pauli_string_embed_sparse(op_map: dict, N: int) -> csr_matrix:
    '''
    Define a sparse Pauli string according to the operator map and total number of qubits.
    Args:
        op_map: A dictionary where keys are site indices (0-indexed) and values are 'X', 'Y', 'Z', or 'I'.
        N: Total number of qubits.
    Returns:
        A sparse matrix representing the Pauli string.
    Example:
        op_map = {0: 'X', 2: 'Z'}
        N = 4
        This will create the operator X I Z I on 4 qubits.
    '''
    tem = csr_matrix(np.eye(1))
    for i in range(N):
        if i in op_map:
            tem = kron(tem, csr_matrix(Pauli(op_map[i])), format='csr')
        else:
            tem = kron(tem, csr_matrix(Pauli('I')), format='csr')
    return tem

def random_Pauli_vec()-> np.ndarray:
    '''
    Return with a normalized Pauli vector i.e. n_x s_x + n_y s_y + n_z s_z
    where n_x^2 + n_y^2 + n_z^2 = 1 and the distribution is uniform on the unit sphere.
    Returns:
        A matrix representing the normalized Pauli vector.
    '''
    phi = np.random.uniform(0, 2 * np.pi)
    cos_theta = np.random.uniform(-1, 1)
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    n_x = sin_theta * np.cos(phi)
    n_y = sin_theta * np.sin(phi)
    n_z = cos_theta
    return n_x * Pauli('X') + n_y * Pauli('Y') + n_z * Pauli('Z')

def random_Pauli_rotation(N: int):
    'Defines an random rotation along some Pauli axis.'
    from .Random import random_bit
    tem = np.eye(1)
    bits = random_bit(2*N)
    for i in range(N):
        if bits[2*i] == 0:
            if bits[2*i+1] ==0:
                tem = np.kron(tem,Pauli('I'))
            else:
                tem = np.kron(tem,Pauli('X'))
        else:
            if bits[2*i+1] ==0:
                tem = np.kron(tem,Pauli('Y'))
            else:
                tem = np.kron(tem,Pauli('Z'))
    return tem

def random_Pauli_vec_multi(N: int)-> np.ndarray:
    '''
    Return a matrix of normalized Pauli vector for N qubits.
    i.e. prod_i (n_x^i s_x^i + n_y^i s_y^i + n_z^i s_z^i)
    Args:
        N: Number of qubits.
    Returns:
        A matrix representing the normalized Pauli vector for N qubits.
    '''
    tem = random_Pauli_vec()
    for i in range(N-1):
        tem = np.kron(tem, random_Pauli_vec())
    return tem

def random_Pauli_vec_rotation(N: int):
    '''
    Define random rotation where each site none trivially supports some pauli vector.
    i.e. expm(2j* pi *rand * prod_i (n_x^i s_x^i + n_y^i s_y^i + n_z^i s_z^i))
    Args:
        N: Number of qubits.
    Returns:
        A unitary matrix representing the random Pauli vector rotation for N qubits.
    '''
    Tem = random_Pauli_vec_multi(N)
    RanNum = np.random.rand()
    Rot = expm(2j * np.pi * RanNum * Tem)
    return Rot

def random_phase_gate() -> np.ndarray:
    'Define a phase gate with random rotation angle.'
    RanNum1 = np.random.rand()
    Rot_Phase = np.array([[np.e**(1j*np.pi*RanNum1),0],[0,np.e**(-1j*np.pi*RanNum1)]])
    return Rot_Phase

def phase_gate(g: float) -> np.ndarray:
    'Define a phase gate with fixed rotation angle: P = Diag(1,e^{ig})'
    tem = np.diagflat(np.exp(np.array([0,1j * g])))
    return tem

def random_Pauli_string_sparse_k_local(N: int, k: int) -> csr_matrix:
    '''
    Define a random sparse Pauli string on N qubits with geometric k-local non-trivial support.
    Args:
        N: Number of qubits.
        k: Number of geometric non-trivial local qubits.
    Returns:
        A sparse matrix representing the random Pauli string on N qubits with k-local support.
    Note: k <= N
    '''
    if k > N:
        raise ValueError("k must be less than or equal to N.")
    start_site = np.random.randint(0, N - k + 1)
    tem = random_Pauli_string(k)
    tem = csr_matrix(tem)
    first_part = eye(2**start_site, format='csr')
    last_part = eye(2**(N - k - start_site), format='csr')
    tem = kron(first_part, tem, format='csr')
    tem = kron(tem, last_part, format='csr')
    return tem

def random_Pauli_string(N: int) -> np.ndarray:
    '''
    Define a random Pauli string on N qubits.
    Args:
        N: Number of qubits.
    Returns:
        A matrix representing the random Pauli string on N qubits.
    '''
    tem = np.eye(1)
    for i in range(N):
        Pauli_choice = ['I', 'X', 'Y', 'Z']
        tem = np.kron(tem, Pauli(np.random.choice(Pauli_choice)))
    return tem