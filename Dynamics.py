'''
Created by Richard Liu, all rights reserved.\n
This module is used to generate some typical functions related to many-body dynamics.\n
Date: Sep 25, 2025
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