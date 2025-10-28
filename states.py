'''
Created by Richard Liu, all rights reserved.\n
This module is used to generate some typical states. However, for the random quantum state, please refer to 'Random' module.\n
Date: Sep 26, 2025
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
    
def partial_trace_keep(rho: np.ndarray, keep: list, dims: list) -> np.ndarray:
    """
    Reduce a multipartite density matrix by tracing out the complement of `keep`.
    I.e., keep the subsystems indexed by `keep`, trace out the others.

    Args:
        rho : (D, D) density matrix with D = prod(dims)
        keep: indices (0..N-1) of subsystems to keep
        dims: list of local dimensions [d0, d1, ..., d_{N-1}]

    Returns:
        rho_keep: reduced density matrix on ⊗_{i in keep} H_i  with shape (prod(d_keep), prod(d_keep))
    """
    N = len(dims)
    D = np.prod(dims)
    if rho.shape != (D, D):
        raise ValueError(f"rho shape {rho.shape} != ({D},{D}) from dims={dims}")

    # sort for deterministic ordering
    keep = sorted(keep)
    all_sys = list(range(N))
    trace_out = [i for i in all_sys if i not in keep]

    dim_keep = int(np.prod([dims[i] for i in keep], dtype=np.int64)) if keep else 1
    dim_out  = int(np.prod([dims[i] for i in trace_out], dtype=np.int64)) if trace_out else 1

    # reshape to 2N indices: (in_0,...,in_{N-1}, out_0,...,out_{N-1})
    rho_nd = rho.reshape([*dims, *dims])

    # permute to (keep_in, out_in, keep_out, out_out)
    perm = keep + trace_out + [i + N for i in keep] + [i + N for i in trace_out]
    rho_perm = np.transpose(rho_nd, axes=perm)

    # group as (dim_keep, dim_out, dim_keep, dim_out)
    rho_grouped = rho_perm.reshape(dim_keep, dim_out, dim_keep, dim_out)

    # partial trace over the 'out' system (the complement), i.e. trace axes 1 and 3
    rho_keep = np.trace(rho_grouped, axis1=1, axis2=3)

    return rho_keep.reshape(dim_keep, dim_keep)

def _psd_frac_power(M: np.ndarray, p:float, tol=1e-12):
    """Return M^p for PSD M via eigen-decomposition; zeros out tiny eigenvalues."""
    w, U = np.linalg.eigh(M)
    w_p = np.where(w > tol, w**p, 0.0)
    return (U * w_p) @ U.conj().T

def make_petz_recovery_operator(rho_B: np.ndarray,
                                rho_BC: np.ndarray,
                                dA: int,
                                tol: float = 1e-12):
    """
    Build the Petz recovery linear map Φ: L(H_A⊗H_B) → L(H_A⊗H_B⊗H_C)
    in the form Φ(X) = (I_A⊗T) X (I_A⊗T)† with T = ρ_BC^{1/2} (ρ_B^{-1/2} ⊗ I_C).

    Parameters
    ----------
    rho_B  : (dB, dB) PSD density matrix on B
    rho_BC : (dB*dC, dB*dC) PSD density matrix on B⊗C
    dA     : dimension of subsystem A
    tol    : eigenvalue cutoff for numerical stability

    Returns
    -------
    A function that applies the Petz recovery map to an input matrix.
    """
    dB = rho_B.shape[0]
    dBC = rho_BC.shape[0]
    if dBC % dB != 0:
        raise ValueError("rho_BC dimension not divisible by dim(B).")
    dC = dBC // dB

    # Precompute fractional powers
    rho_B_inv_sqrt = _psd_frac_power(rho_B, -0.5, tol)
    rho_BC_sqrt    = _psd_frac_power(rho_BC, 0.5, tol)

    rho_BC_sqrt = rho_BC_sqrt.reshape(dB, dC, dB, dC)
    T = np.einsum('ijkl,km->ijml', rho_BC_sqrt, rho_B_inv_sqrt)
    T_conj = T.reshape(dBC,dBC).conj().T
    T_conj = T_conj.reshape(dB,dC,dB,dC)

    # Define the linear map
    def apply(X_AB):
        # X_AB: (dA*dB, dA*dB)
        X_AB = X_AB.reshape(dA, dB, dA, dB)  # (a,b,a',b')
        # Apply T @ X_AB @ T† on the B indices
        temp = np.einsum('ijml,bmac-> ijblac', T, X_AB)
        temp = np.einsum('ijblac, clno -> bijano', temp, T_conj)
        rho_ABC = temp.reshape(dA*dBC, dA*dBC)
        return rho_ABC
    
    return apply
    
def Petz_recovery(rho_AB: np.ndarray, rho_B: np.ndarray, rho_BC: np.ndarray) -> np.ndarray:
    '''
    Perform the Petz recovery map on a tripartite quantum state.
    rho_ABC = rho_{BC}^{1/2} rho_{B}^{-1/2} (rho_{AB}) rho_{B}^{-1/2} rho_{BC}^{1/2}
    Args:
        rho_AB: The density matrix of subsystem AB as a numpy array.
        rho_B: The density matrix of subsystem B as a numpy array.
        rho_BC: The density matrix of subsystem BC as a numpy array.
    Returns:
        The recovered density matrix of subsystem ABC as a numpy array.
    '''
    # Eigen-decomposition of rho_B
    evals, evecs = np.linalg.eigh(rho_B)
    # Inverse square root of rho_B
    evals_inv_sqrt = np.diag([1/np.sqrt(ev) if ev > 1e-12 else 0 for ev in evals])
    rho_B_inv_sqrt = evecs @ evals_inv_sqrt @ evecs.conj().T
    
    # Square root of rho_BC
    evals_BC, evecs_BC = np.linalg.eigh(rho_BC)
    evals_BC_sqrt = np.diag([np.sqrt(ev) if ev > 1e-12 else 0 for ev in evals_BC])
    rho_BC_sqrt = evecs_BC @ evals_BC_sqrt @ evecs_BC.conj().T

    # Dimensions
    dA = rho_AB.shape[0] // rho_B.shape[0]
    dB = rho_B.shape[0]
    dC = rho_BC.shape[1] // dB
    

    # Petz recovery map from AB and BC to ABC
    # rho_ABC = rho_{BC}^{1/2} rho_{B}^{-1/2} (rho_{AB}) rho_{B}^{-1/2} rho_{BC}^{1/2}
    rho_AB = rho_AB.reshape(dA, dB, dA, dB)
    rho_BC_sqrt = rho_BC_sqrt.reshape(dB, dC, dB, dC)
    # First apply rho_B^{-1/2} on the B indices of rho_AB
    temp = np.einsum('ij,kjlm->kilm', rho_B_inv_sqrt, rho_AB)
    temp = np.einsum('ijkl,lm->ijkm', temp, rho_B_inv_sqrt)
    # Then apply rho_BC^{1/2} on the B indices of the result
    temp = np.einsum('ijkl,mkno->ijmlno', rho_BC_sqrt, temp)
    rho_ABC = np.einsum('ijmlno,olst->mijnst', temp, rho_BC_sqrt)
    # Reshape back to matrix form
    rho_ABC = rho_ABC.reshape(dA * dB * dC, dA * dB * dC)
    return rho_ABC
    