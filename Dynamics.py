'''
Created by Richard Liu, all rights reserved.\n
This module is used to generate some typical functions related to many-body dynamics.\n
Date: Sep 18, 2025
'''

import numpy as np

def correlation_inf_tem(H:np.ndarray, TimeList:list, op: np.ndarray) -> np.ndarray:
    '''
    Calculate the correlation function at infinite temperature using exact diagonalization.
    Args:
        H: The Hamiltonian of the system as a numpy array.
        TimeList: A list of time points at which to evaluate the correlation function.
        op: The operator whose correlation function is to be calculated as a numpy array.
    Returns:
        A numpy array representing the correlation function at each time point in TimeList.
    '''
    # Diagonalize H
    E,V = np.linalg.eigh(H)
    Vh = V.conj().T
    # Rotate O into eigenbasis and square magnitudes
    O_tilde = Vh @ op @ V
    W = np.abs(O_tilde)**2     # | \tilde O_{mn} |^2
    # Vectorized evaluation for all t
    C = np.zeros(len(TimeList), dtype=np.float64)
    d = H.shape[0]
    for i in range(len(TimeList)):
        t = TimeList[i]
        p = np.exp(1j * E * t)               # (d,)
        C[i] = (np.vdot(p, W @ p)).real / d  # <O(0)O(t)> = Tr[O(t)O(0)]/2^N
    return C