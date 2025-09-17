'''
Created by Richard Liu, all rights reserved.
Date: Sep 17, 2025
'''
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix,kron
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

def random_Pauli_vec():
    'Return with a normalized Pauli vector i.e. n_x s_x + n_y s_y + n_z s_z'
    phi = np.random.uniform(0, 2 * np.pi)
    u = np.random.uniform(0, 1)
    theta = np.arccos(2 * u - 1)
    vec = Pauli('X') * np.sin(theta) * np.cos(phi) + Pauli('Y') * np.sin(theta) * np.sin(phi) + Pauli('Z') * np.cos(theta)
    return vec

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

def random_Pauli_vec_rotation(N: int):
    '''
    Define random rotation where each site none trivially supports some pauli vector.

    i.e. expm(2j* pi *rand * \prod n_x s_x + n_y s_y + n_z s_z)
    '''
    tem = np.eye(1)
    for i in range(N):
        tem = np.kron(tem,random_Pauli_vec())
    theta = np.random.rand()
    rotate = expm(2j * np.pi * theta * tem)
    return rotate

def random_phase_gate() -> np.ndarray:
    'Define a phase gate with random rotation angle.'
    RanNum1 = np.random.rand()
    Rot_Phase = np.array([[np.e**(1j*np.pi*RanNum1),0],[0,np.e**(-1j*np.pi*RanNum1)]])
    return Rot_Phase

def phase_gate(g: float) -> np.ndarray:
    'Define a phase gate with fixed rotation angle: P = Diag(1,e^{ig})'
    tem = np.diagflat(np.exp(np.array([0,1j * g])))
    return tem