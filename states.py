'''
Created by Richard Liu, all rights reserved.\n
This module is used to generate some typical states. However, for the random quantum state, please refer to 'Random' module.\n
Date: Apr 11, 2024
'''
import numpy as np

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