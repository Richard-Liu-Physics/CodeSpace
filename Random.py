'''
Created by Richard Liu, all rights reserved.
Date: Oct 10, 2024
'''
import numpy as np

def random_Haar_unitary(N: int):
    '''
    Define a random haar unitary for N qubits.\n
    Return value is a numpy matrix of shape (2**N, 2**N)
    '''
    Z = np.random.randn(2**N,2**N) + 1j * np.random.randn(2**N,2**N)
    Q, R = np.linalg.qr(Z)
    D = np.diag(np.diag(R) / np.abs(np.diag(R)))
    U = np.dot(Q, D)
    return U

def random_haar_ket(N: int):
    '''
    Define a random haar state for N qubits with shape (2**N,)
    '''
    Z = np.random.randn(2**N) + 1j * np.random.randn(2**N)
    Z = Z / np.linalg.norm(Z)
    return Z

def rand_product_ket(N: int):
    '''
    Define a random product state on N qubits with shape (2**N,). On each site we create a random haar state.
    '''
    Z = random_haar_ket(1)
    for z in range(1,N):
        Ztem = random_haar_ket(1)
        Z = np.kron(Z,Ztem)
    return Z

def random_GUE(N: int):
    '''
    Define a random GUE matrix for N qubits with shape (2**N, 2**N).
    '''
    random_matrix = np.random.randn(2**N, 2**N) + 1j * np.random.randn(2**N, 2**N)
    GUE_matrix = (random_matrix + random_matrix.conj().T) / 2
    return GUE_matrix

def random_GOE(N: int):
    '''
    Define a random GOE matrix for N qubits with shape (2**N, 2**N).
    '''
    random_matrix = np.random.randn(2**N, 2**N)
    GOE_matrix = (random_matrix + random_matrix.T) / 2
    return GOE_matrix

def random_GUE_normalised(N: int):
    '''
    Define a random GUE matrix for N qubits with shape (2**N, 2**N) and make it normalized in trace norm.
    '''
    random_matrix = random_GUE(N)
    norm = np.sqrt(np.trace(random_matrix @ random_matrix.conj().T))
    random_matrix = random_matrix/norm
    return random_matrix

def random_bit(N: int):
    'Return with a N-bit random bit string.'
    bit_list = []
    for i in range(N):
        if np.random.rand(1)[0] < 1/2:
            bit_list.append(0)
        else:
            bit_list.append(1)
    return bit_list