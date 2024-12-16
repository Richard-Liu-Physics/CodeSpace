'''
Created by Richard Liu, all rights reserved.\n
This module is used to generate some typical states. However, for the random quantum state, please refer to 'Random' module.\n
Date: Oct 18, 2024
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