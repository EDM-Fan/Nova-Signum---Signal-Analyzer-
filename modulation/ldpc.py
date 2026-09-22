"""
modulation/ldpc.py
==================
Low-Density Parity-Check (LDPC) Forward Error Correction (FEC) module.

Features:
- Deterministic rate-1/2 (n=128, k=64) regular accumulator / IRA LDPC code
  (dv=3 variable node column weight, dual-diagonal low-density parity structure).
- Systematic generator matrix G = [I_k | P^T] ensuring exact information bit preservation.
- Fast Normalized Min-Sum iterative Belief-Propagation (BP) decoder.
- Support for both hard-decision binary bits {0, 1} and soft-decision LLR inputs.
- Clean convergence detection via syndrome check H @ c^T == 0 (mod 2).
"""

from __future__ import annotations
import numpy as np


def build_ldpc_code(n: int = 128, k: int = 64, dv: int = 3, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct a deterministic rate-1/2 (n, k) LDPC parity check matrix H and
    its systematic generator matrix G.
    
    Structure:
    - H has shape (m, n) where m = n - k.
    - H is partitioned as [H_u | H_p], where:
      * H_u is (m, k) with regular column weight dv=3.
      * H_p is (m, m) lower dual-diagonal (accumulator structure), invertible over GF(2).
    - Generator matrix G = [I_k | (H_p^{-1} @ H_u)^T] of shape (k, n).
    - Guarantees H @ G^T == 0 (mod 2).
    - Systematic property: codeword c = u @ G = [u | p], so c[:k] == u.
    """
    m = n - k
    rng = np.random.default_rng(seed)
    
    # Dual-diagonal parity structure (lower bidiagonal, det(H_p) = 1 over GF2)
    Hp = np.eye(m, dtype=np.uint8)
    for r in range(1, m):
        Hp[r, r - 1] = 1
        
    # Construct H_u of shape (m, k) with regular column degree dv
    Hu = np.zeros((m, k), dtype=np.uint8)
    row_counts = np.zeros(m, dtype=int)
    
    for j in range(k):
        chosen = []
        for _ in range(dv):
            cands = [r for r in range(m) if r not in chosen]
            # Prioritize rows with lower degree for uniform check distribution
            deg_order = np.argsort([row_counts[r] for r in cands])
            top_pool = [cands[idx] for idx in deg_order[:min(10, len(cands))]]
            r_pick = rng.choice(top_pool)
            chosen.append(r_pick)
            row_counts[r_pick] += 1
        for r in chosen:
            Hu[r, j] = 1
            
    H = np.hstack([Hu, Hp])
    
    # Invert lower-bidiagonal H_p: H_p^{-1} is lower-triangular all ones
    Hp_inv = np.tri(m, dtype=np.uint8)
    P_T = (Hp_inv @ Hu).T % 2
    G = np.hstack([np.eye(k, dtype=np.uint8), P_T])
    
    # Sanity verification
    assert np.all((H @ G.T) % 2 == 0), "Parity check matrix orthogonality violated!"
    return H, G


# Default (128, 64) code instance precomputed for efficiency
DEFAULT_H_128_64, DEFAULT_G_128_64 = build_ldpc_code(n=128, k=64, dv=3, seed=42)
_DEFAULT_ROWS, _DEFAULT_COLS = np.where(DEFAULT_H_128_64 == 1)
_DEFAULT_NUM_EDGES = len(_DEFAULT_ROWS)
_DEFAULT_C_EDGES = [np.where(_DEFAULT_ROWS == c)[0] for c in range(DEFAULT_H_128_64.shape[0])]
_DEFAULT_V_EDGES = [np.where(_DEFAULT_COLS == v)[0] for v in range(DEFAULT_H_128_64.shape[1])]


def ldpc_encode(info_bits: np.ndarray, H: np.ndarray | None = None, G: np.ndarray | None = None) -> np.ndarray:
    """
    Systematically encode information bits into an LDPC codeword.
    
    Parameters:
        info_bits: 1D array of binary bits {0, 1} with length k (or multiple of k).
        H: (m, n) parity-check matrix (optional if G is provided or default used).
        G: (k, n) systematic generator matrix. If None, default (128, 64) code is used.
        
    Returns:
        codeword_bits: 1D np.ndarray of encoded bits {0, 1} of length n (or multiple of n).
    """
    if G is None:
        if H is not None and (H.shape[0] != DEFAULT_H_128_64.shape[0] or H.shape[1] != DEFAULT_H_128_64.shape[1]):
            raise ValueError("Custom H provided without corresponding G matrix")
        G = DEFAULT_G_128_64
        
    k, n = G.shape
    u = np.asarray(info_bits, dtype=np.uint8).ravel()
    
    if len(u) == 0:
        return np.array([], dtype=np.uint8)
        
    # Handle arbitrary lengths: pad or process block-by-block
    if len(u) % k != 0:
        pad_len = k - (len(u) % k)
        u = np.pad(u, (0, pad_len), mode="constant")
        
    num_blocks = len(u) // k
    codewords = []
    for blk in range(num_blocks):
        u_blk = u[blk * k : (blk + 1) * k]
        cw_blk = (u_blk @ G) % 2
        codewords.append(cw_blk)
        
    return np.concatenate(codewords).astype(np.uint8)


def ldpc_decode(
    received: np.ndarray,
    H: np.ndarray | None = None,
    max_iterations: int = 50,
    alpha: float = 0.8,
    raise_on_failure: bool = False,
) -> tuple[np.ndarray, bool, int]:
    """
    Decode received bits or LLRs using the Normalized Min-Sum Belief-Propagation algorithm.
    
    Parameters:
        received: 1D array of received bits {0, 1} or soft LLRs (length multiple of n).
        H: (m, n) parity-check matrix (default: (64, 128) matrix).
        max_iterations: Maximum belief-propagation decoding iterations.
        alpha: Min-Sum normalization scaling factor (default: 0.8 for high stability).
        raise_on_failure: If True, raises ValueError when parity checks fail to converge.
        
    Returns:
        (decoded_info_bits, converged, iterations):
            decoded_info_bits: 1D np.ndarray of recovered information bits {0, 1}.
            converged: bool, True if all parity check constraints were satisfied.
            iterations: int, number of iterations executed.
    """
    if H is None:
        H = DEFAULT_H_128_64
        
    m, n = H.shape
    k = n - m
    
    rx = np.asarray(received, dtype=float).ravel()
    if len(rx) == 0:
        return np.array([], dtype=np.uint8), True, 0
        
    if len(rx) % n != 0:
        pad_len = n - (len(rx) % n)
        rx = np.pad(rx, (0, pad_len), mode="constant", constant_values=0.0)
        
    # Precompute or retrieve edge lists
    if H is DEFAULT_H_128_64 or H is None:
        rows, cols = _DEFAULT_ROWS, _DEFAULT_COLS
        num_edges = _DEFAULT_NUM_EDGES
        c_edges, v_edges = _DEFAULT_C_EDGES, _DEFAULT_V_EDGES
    else:
        rows, cols = np.where(H == 1)
        num_edges = len(rows)
        c_edges = [np.where(rows == c)[0] for c in range(m)]
        v_edges = [np.where(cols == v)[0] for v in range(n)]
    
    num_blocks = len(rx) // n
    decoded_blocks = []
    all_converged = True
    total_iters = 0
    
    for blk in range(num_blocks):
        rx_blk = rx[blk * n : (blk + 1) * n]
        
        # Convert binary bits {0, 1} to initial channel LLRs (+6.0 for 0, -6.0 for 1)
        if np.all(np.isin(rx_blk, [0, 1])):
            llr_ch = np.where(rx_blk == 0, 6.0, -6.0)
        else:
            llr_ch = rx_blk.copy()
            
        # Check if already a valid codeword
        hard_initial = (llr_ch < 0).astype(np.uint8)
        if np.all((H @ hard_initial) % 2 == 0):
            decoded_blocks.append(hard_initial[:k])
            total_iters += 0
            continue
            
        # Message passing edge initialization
        q_edges = llr_ch[cols].copy()
        r_edges = np.zeros(num_edges, dtype=float)
        
        blk_converged = False
        blk_iters = 0
        hard_bits = hard_initial
        
        for it in range(1, max_iterations + 1):
            blk_iters = it
            
            # 1. Check Node Update (along each check node's incident edges)
            for c in range(m):
                edges = c_edges[c]
                q_vals = q_edges[edges]
                signs = np.sign(q_vals)
                signs[signs == 0] = 1.0
                abs_vals = np.abs(q_vals)
                
                # Min1, Min2 finding for Min-Sum
                min1_idx = np.argmin(abs_vals)
                min1 = abs_vals[min1_idx]
                abs_vals_temp = abs_vals.copy()
                abs_vals_temp[min1_idx] = np.inf
                min2 = np.min(abs_vals_temp)
                
                total_sign = np.prod(signs)
                
                for i, e in enumerate(edges):
                    s = total_sign * signs[i]
                    m_val = min2 if i == min1_idx else min1
                    r_edges[e] = alpha * s * m_val
                    
            # 2. Variable Node Update
            L_tot = llr_ch.copy()
            for v in range(n):
                edges = v_edges[v]
                L_tot[v] += np.sum(r_edges[edges])
                for e in edges:
                    q_edges[e] = L_tot[v] - r_edges[e]
                    
            # 3. Syndrome check
            hard_bits = (L_tot < 0).astype(np.uint8)
            if np.all((H @ hard_bits) % 2 == 0):
                blk_converged = True
                break
                
        if not blk_converged:
            all_converged = False
            
        total_iters += blk_iters
        decoded_blocks.append(hard_bits[:k])
        
    decoded_info = np.concatenate(decoded_blocks).astype(np.uint8)
    
    if raise_on_failure and not all_converged:
        raise ValueError(
            f"LDPC decoding failed: parity checks not satisfied after {max_iterations} iterations."
        )
        
    return decoded_info, all_converged, total_iters
