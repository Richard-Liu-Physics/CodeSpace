'''
Created by Richard Liu, all rights reserved.
Date: Jan 26, 2026
'''
import numpy as np
import qutip as qt
import tqdm.auto as tqdm

def SWAP_Operator(N: int):
    'Construct the SWAP operator for two N-dimensional systems.'
    Swap = np.zeros((N,N,N,N))
    for i in range(N):
        for j in range(N):
            Swap[i][j][j][i] = 1
    Swap = Swap.reshape((N**2,N**2))
    return qt.Qobj(Swap)



def Haar_Density(N: int, k: int):
    'Calculate the k-copied density matrix for a Haar random state; N is the number of dimension'
    if k == 1:
        return qt.Qobj(np.eye(N)/ N)
    elif k == 2:
        Id = np.identity(N**2)
        Swap = SWAP_Operator(N).data.toarray()
        tem = (Id+Swap)/(N+1)/N
        return qt.Qobj(tem)

def tracedist(state_list: list, k: int, D: int, timebar = True):
    '''Calculate the k-th trace distance - time evolution for the temporal ensemble. The list should contains (D,) vectors.\n
    The default timebar parameter gives a time processing bar relying on tqdm, when it's set to False, there is no time processing bar.'''
    dif_list = []
    Haar = Haar_Density(D,k)
    Den_ave = np.zeros((D**k,D**k))
    if timebar == True:
        for i in tqdm.range(len(state_list), desc="Calculating Trace Distance", leave=False, position=len(tqdm._instances)):
            Den_ave = Den_ave * i/(i+1)
            tem = np.outer(state_list[i],state_list[i].conj().T)
            Density_Matrix = np.eye(1)
            for j in range(k):
                Density_Matrix = np.kron(Density_Matrix, tem)
            Density_Matrix = Density_Matrix/(i+1)

            Den_ave = Den_ave + Density_Matrix

            # Cauculate the trace-dist
            Den_ave_copy = qt.Qobj(Den_ave)
            dif = qt.tracedist(Den_ave_copy,Haar)
            dif_list.append(dif)
    else:
        for i in range(len(state_list)):
            Den_ave = Den_ave * i/(i+1)
            tem = np.outer(state_list[i],state_list[i].conj().T)
            Density_Matrix = np.eye(1)
            for j in range(k):
                Density_Matrix = np.kron(Density_Matrix, tem)
            Density_Matrix = Density_Matrix/(i+1)

            Den_ave = Den_ave + Density_Matrix

            # Cauculate the trace-dist
            Den_ave_copy = qt.Qobj(Den_ave)
            dif = qt.tracedist(Den_ave_copy,Haar)
            dif_list.append(dif)
    
    return dif_list

def Frame_Potential(state_list: list, k: int, D: int):
    'Calculate the k-th frame potential - time evolution for the temporal ensemble. The list should contains (D,) vectors.'
    print("Under construction")
    return None

def Unitary_2_Design(ensemble: list, D: int):
    'Calculate the 2-design condition for the given ensemble of unitaries. The ensemble should contains (D,D) matrices.'
    Id = np.identity(D**2)
    Swap = np.zeros((D,D,D,D))
    for i in range(D):
        for j in range(D):
            Swap[i][j][j][i] = 1
    Swap = Swap.reshape((D**2,D**2))
    Haar = (Id+Swap)/(D+1)/D

    Den_ave = np.zeros((D**2,D**2))
    for i in range(len(ensemble)):
        U = ensemble[i]
        U_conj = U.conj().T
        Density_Matrix = np.kron(U,U_conj)
        Density_Matrix = Density_Matrix/(len(ensemble))

        Den_ave = Den_ave + Density_Matrix

    Den_ave_copy = qt.Qobj(Den_ave)
    dif = qt.norm(Den_ave_copy - Haar)
    return dif