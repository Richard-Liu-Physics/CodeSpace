'''
Created by Richard Liu, all rights reserved.\n
This module is used to generate some typical functions related to many-body dynamics.\n
Date: Jan 26, 2026
'''

import numpy as np
from tqdm.auto import tqdm

def correlation_inf_temp(H:np.ndarray, TimeList:list, ops: list) -> list:
    '''
    Calculate the correlation function at infinite temperature using exact diagonalization.
    Args:
        H: The Hamiltonian of the system as a numpy array.
        TimeList: A list of time points at which to evaluate the correlation function.
        ops: A list of operators whose correlation function is to be calculated, each as a numpy array.
    Returns:
        A list of numpy arrays representing the correlation functions at each time step for each operator.
    '''
    # Diagonalize H
    E,V = np.linalg.eigh(H)
    Vh = V.conj().T
    
    # Prepare output
    ans = []
    # Rotate O into eigenbasis and square magnitudes
    for op in ops:
        O_tilde = Vh @ op @ V
        W = np.abs(O_tilde)**2     # | \tilde O_{mn} |^2
        # Vectorized evaluation for all t
        C = np.zeros(len(TimeList), dtype=np.float64)
        d = H.shape[0]
        for i in tqdm(range(len(TimeList)), desc="Calculating Correlation", leave=False, position=len(tqdm._instances)):
            t = TimeList[i]
            p = np.exp(1j * E * t)               # (d,)
            C[i] = (np.vdot(p, W @ p)).real / d  # <O(0)O(t)> = Tr[O(t)O(0)]/2^N
        ans.append(C)
    return ans

def gate_2site(i: int, j: int, psi: np.ndarray, U: np.ndarray) -> np.ndarray:
    U = U.reshape(2, 2, 2, 2)  # U[a,b,c,d] : (in a,b) -> (out c,d)

    if i < j:
        psi = psi.reshape([2**i, 2, 2**(j-i-1), 2, -1])
        psi = np.einsum('iajbk, abcd -> icjdk', psi, U)
    else:
        # embed on (j,i), but we want the logical ordering (i,j)
        # so conjugate by SWAP at the tensor level: U'_{a b c d} = U_{b a d c}
        U_swapped = U.transpose(1, 0, 3, 2)  # (b,a,d,c)
        psi = psi.reshape([2**j, 2, 2**(i-j-1), 2, -1])
        psi = np.einsum('iajbk, abcd -> icjdk', psi, U_swapped)

    return psi.reshape(-1)

def apply_1site_gate(i: int, psi: np.ndarray, U: np.ndarray) -> np.ndarray:
    '''
    Apply a single-site gate to site i in an N-site system.
    Args:
        i: The site index (0-based).
        psi: The state vector of the N-site system as a numpy array.
        U: The single-site gate as a 2x2 numpy array.
    Returns:
        The new state vector after applying the single-site gate.
    '''
    psi = psi.reshape([2**i, 2, -1])
    psi = np.einsum('iak, ab -> ibk', psi, U)
    psi = psi.reshape([-1])
    return psi