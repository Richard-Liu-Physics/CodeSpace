'''
Created by Richard Liu, all rights reserved.
Date: Jan 26, 2026
'''
import numpy as np
from copy import deepcopy
from scipy.stats import entropy
def RDM(n: float, l: int, N: int, State: np.ndarray) -> np.ndarray:
    'Calculate the reduced density matrix within the interval (n, l). Note that the index we adapt starts from 0. The system size is N.'
    start = round(n - l/2)
    stop = round(n + l/2)
    NewOrder = [2**start,2**(stop-start+1),2**(N-stop-1)]
    state = deepcopy(State)
    state = state.reshape(NewOrder)
    stateconj = state.conj()
    R = np.einsum('ijk,imk ->jm', state, stateconj)
    return R

def RDM_smaller(n: float, l: int, N: int, State: np.ndarray) -> np.ndarray:
    '''
    Calculate the reduced density matrix within the interval (n, l). Note that the index we adapt starts from 0. The system size is N.
    
    If l is larger than half of the system size, return the reduced density matrix of its complement.
    '''
    start = round(n - l/2)
    stop = round(n + l/2)
    NewOrder = [2**start,2**(stop-start+1),2**(N-stop-1)]
    state = deepcopy(State)
    state = state.reshape(NewOrder)
    stateconj = state.conj()
    if l < N/2:
        R = np.einsum('ijk,imk ->jm', state, stateconj)
    else:
        R = np.einsum('ijk , ljm ->iklm', state, stateconj)
        R = R.reshape(2**(N-l-1),2**(N-l-1))
    return R
    
def voninfo(l: int, Reduced_density_matrix: np.ndarray) -> float:
    'Calculate the total von Neumann information within interval of length l.'
    Eig = np.linalg.eigvalsh(Reduced_density_matrix)
    Eig = np.maximum(Eig, 0)
    Entropy = entropy(Eig)
    voninf = np.log(2**(l+1)) - Entropy
    return voninf

def infdot(State: np.ndarray, N: int):
    'Calculate the information in the information lattice. Return with a list with the element type [n,l,information].'
    dot_map = np.zeros((2*N,N)) # [2*n,l,total information within interval]
    dot_return = []
    for l in range(N):
        start = l/2
        for delta in range(N - l):
            n = delta + start
            n_map = round(2*n)
            R = RDM_smaller(n, l, N, State)
            inf_tem = voninfo(l,R)
            dot_map[n_map][l] = inf_tem

    for l in range(N):
        start = l/2
        for delta in range(N - l):
            n = delta + start
            n_map = round(2*n)
            if l == 0:
                dot_return.append([n,l,dot_map[n_map][l]])
            elif l == 1:
                tem = dot_map[n_map][1] - dot_map[n_map - 1][0] - dot_map[n_map + 1][0]
                dot_return.append([n,l,tem])
            else:
                tem = dot_map[n_map][l] - dot_map[n_map - 1][l -1] - dot_map[n_map + 1][l - 1] + dot_map[n_map][l - 2]
                dot_return.append([n,l,tem])
    return dot_return
        