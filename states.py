'''
Created by Richard Liu, all rights reserved.\n
This module is used to generate some typical states. However, for the random quantum state, please refer to 'Random' module.\n
Date: Sep 17, 2025
'''
import numpy as np
import scipy.sparse as sp

def GHZ(N: int) -> np.ndarray:
    'Generate a N-qubit GHZ state'
    tem = np.zeros(2**N)
    tem[0] = 1/np.sqrt(2)
    tem[2**N - 1] = 1/np.sqrt(2)
    return tem

def W(N: int) -> np.ndarray:
    'Generate a N-qubit W state'
    tem = np.zeros(2**N)
    amplitude = 1 / np.sqrt(N)
    for i in range(N):
        tem[2**i] = amplitude
    return tem

def tilted_sigle(theta: float, phi:float) -> np.ndarray:
    'Generate a tilted single qubit state'
    tem = np.zeros(2, dtype=complex)
    tem[0] = np.cos(theta/2)
    tem[1] = np.exp(1j * phi) * np.sin(theta/2)
    return tem

def product_state(state_list: list) -> np.ndarray:
    'Generate a product state from a list of single qubit states'
    tem = state_list[0]
    for state in state_list[1:]:
        tem = np.kron(tem, state)
    return tem

def expectation_value_real(state: np.ndarray, op: np.ndarray) -> float:
    'Calculate the expectation value of an operator with respect to a given state. The state can be a density matrix.'
    if len(state.shape) == 1:
        bra = np.conjugate(state).T
        ket = state
        return np.real(bra @ op @ ket)
    elif len(state.shape) == 2:
        return np.real(np.trace(state @ op))
    else:
        raise ValueError("State must be a vector or a density matrix.")