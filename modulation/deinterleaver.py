"""
modulation/deinterleaver.py
===========================
Interleaving / de-interleaving schemes for burst-error dispersion.

Provided schemes
----------------
1. Block (R×C matrix)         -- write rows, read columns.
2. Convolutional (Forney)     -- depth-D shift-register branches.
3. Pseudo-Random Diagonal     -- seeded PRNG permutation per block.
4. True Diagonal (R×C matrix) -- deterministic cyclic-shift read-out
                                  along diagonal bands; no seed/randomness.
"""

from __future__ import annotations
import numpy as np


def block_interleave(bits: np.ndarray | list[int], rows: int = 8, cols: int = 32) -> np.ndarray:
    """
    Block-interleave input bits across (rows x cols) blocks.

    Parameters
    ----------
    bits : array-like (0s and 1s)
        Input bit stream.
    rows : int
        Number of matrix rows (interleaving depth, default: 8).
    cols : int
        Number of matrix columns (default: 32).

    Returns
    -------
    np.ndarray (1D int array of length N_blocks * rows * cols)
    """
    bits = np.asarray(bits, dtype=int)
    block_size = rows * cols

    n_pad = (block_size - (len(bits) % block_size)) % block_size
    if n_pad > 0:
        padded = np.concatenate([bits, np.zeros(n_pad, dtype=int)])
    else:
        padded = bits

    # Shape: (N_blocks, rows, cols) -> written row-by-row
    # Transpose to (N_blocks, cols, rows) -> read column-by-column
    interleaved = padded.reshape(-1, rows, cols).transpose(0, 2, 1).reshape(-1)
    return interleaved


def block_deinterleave(
    bits: np.ndarray | list[int],
    rows: int = 8,
    cols: int = 32,
    original_len: int | None = None,
) -> np.ndarray:
    """
    De-interleave bits from (rows x cols) blocks back to original order.

    Parameters
    ----------
    bits : array-like (0s and 1s)
        Interleaved bit stream.
    rows : int
        Number of matrix rows (default: 8).
    cols : int
        Number of matrix columns (default: 32).
    original_len : int | None
        If given, slice output to this length to remove trailing pad bits.

    Returns
    -------
    np.ndarray (1D int array)
    """
    bits = np.asarray(bits, dtype=int)
    block_size = rows * cols

    n_pad = (block_size - (len(bits) % block_size)) % block_size
    if n_pad > 0:
        padded = np.concatenate([bits, np.zeros(n_pad, dtype=int)])
    else:
        padded = bits

    # Shape: (N_blocks, cols, rows) -> written column-by-column
    # Transpose to (N_blocks, rows, cols) -> read row-by-row
    deinterleaved = padded.reshape(-1, cols, rows).transpose(0, 2, 1).reshape(-1)

    if original_len is not None:
        return deinterleaved[:original_len]
    return deinterleaved


# ---------------------------------------------------------------------------
# True Diagonal Interleaver (deterministic, no seed/PRNG)
# ---------------------------------------------------------------------------
#
# Classic diagonal read-out pattern:
#   1. Write bits sequentially into an R×C matrix (row-by-row).
#   2. Cyclically shift row r LEFT by r positions.
#   3. Read out column-by-column.
#
# Inverse reverses steps 2–3:
#   1. Write received bits column-by-column into R×C matrix.
#   2. Cyclically shift row r RIGHT by r positions.
#   3. Read row-by-row.
#
# Burst dispersion: a burst of B consecutive output bits maps to at most
# ceil(B/cols) consecutive bits per row in the original stream.
# No seed or PRNG is used — fully deterministic from geometry alone.
#
# Default geometry: rows=16, cols=16 (256-bit blocks, same as pseudo-random
# scheme for fair comparison).
# ---------------------------------------------------------------------------


def true_diagonal_interleave(
    bits: np.ndarray | list[int],
    rows: int = 16,
    cols: int = 16,
) -> np.ndarray:
    """
    True deterministic diagonal interleaver (no seed / no PRNG).

    Writes bits row-by-row into an (rows x cols) matrix, cyclically shifts
    each row r left by r columns, then reads out column-by-column.

    Parameters
    ----------
    bits : array-like (0s and 1s)
        Input bit stream.
    rows : int
        Matrix rows (default: 16).
    cols : int
        Matrix columns (default: 16).

    Returns
    -------
    np.ndarray (1D int array of length N_blocks * rows * cols)
    """
    bits = np.asarray(bits, dtype=int)
    block_size = rows * cols
    n_pad = (block_size - (len(bits) % block_size)) % block_size
    padded = np.concatenate([bits, np.zeros(n_pad, dtype=int)]) if n_pad else bits

    mat = padded.reshape(-1, rows, cols)          # (N, R, C)
    r_idx = np.arange(rows)
    # Column indices after cyclic shift of row r by -r:
    col_idx = (np.arange(cols)[None, :] + r_idx[:, None]) % cols   # (R, C)
    shifted = mat[:, r_idx[:, None], col_idx]                       # (N, R, C)
    # Read column-by-column: (N, C, R) -> flatten
    return shifted.transpose(0, 2, 1).reshape(-1)


def true_diagonal_deinterleave(
    bits: np.ndarray | list[int],
    rows: int = 16,
    cols: int = 16,
    original_len: int | None = None,
) -> np.ndarray:
    """
    Inverse of :func:`true_diagonal_interleave`.

    Writes received bits column-by-column into (rows x cols) matrix,
    cyclically shifts each row r RIGHT by r columns, then reads row-by-row.

    Parameters
    ----------
    bits : array-like (0s and 1s)
        Interleaved bit stream.
    rows : int
        Matrix rows used during interleaving (default: 16).
    cols : int
        Matrix columns used during interleaving (default: 16).
    original_len : int | None
        If given, slice output to this length to remove trailing pad bits.

    Returns
    -------
    np.ndarray (1D int array)
    """
    bits = np.asarray(bits, dtype=int)
    block_size = rows * cols
    n_pad = (block_size - (len(bits) % block_size)) % block_size
    padded = np.concatenate([bits, np.zeros(n_pad, dtype=int)]) if n_pad else bits

    # Received was: (N, C, R) -> we need (N, R, C) first
    mat = padded.reshape(-1, cols, rows).transpose(0, 2, 1)   # (N, R, C)
    r_idx = np.arange(rows)
    # Inverse: shift right by r => col index (c - r) % cols
    col_idx = (np.arange(cols)[None, :] - r_idx[:, None]) % cols   # (R, C)
    recovered = mat[:, r_idx[:, None], col_idx]                     # (N, R, C)
    out = recovered.reshape(-1)
    if original_len is not None:
        return out[:original_len]
    return out


def conv_interleave(
    bits: np.ndarray | list[int],
    depth: int = 8,
    span: int = 4,
    flush: bool = True,
) -> np.ndarray:
    """
    Forney (Ramsey Type III) convolutional interleaver.

    Distributes sequential input bits across `depth` shift-register branches,
    where branch k introduces a delay of k * span bits.

    Parameters
    ----------
    bits : array-like (0s and 1s)
        Input bit stream.
    depth : int
        Number of parallel shift-register branches (default: 8).
    span : int
        Delay increment per branch (default: 4 bits).
    flush : bool
        If True, appends depth * (depth - 1) * span zero flush bits so that
        all input bits emerge from the shift registers.

    Returns
    -------
    np.ndarray (1D int array)
    """
    bits = np.asarray(bits, dtype=int)
    fifos = [[0] * (k * span) for k in range(depth)]

    total_delay = depth * (depth - 1) * span
    if flush:
        in_bits = np.concatenate([bits, np.zeros(total_delay, dtype=int)])
    else:
        in_bits = bits

    out_bits = []
    for i, b in enumerate(in_bits):
        branch = i % depth
        if fifos[branch]:
            out_bits.append(fifos[branch].pop(0))
            fifos[branch].append(b)
        else:
            out_bits.append(b)

    return np.array(out_bits, dtype=int)


def conv_deinterleave(
    bits: np.ndarray | list[int],
    depth: int = 8,
    span: int = 4,
    trim_delay: bool = True,
    original_len: int | None = None,
) -> np.ndarray:
    """
    Complementary Forney convolutional de-interleaver (exact mathematical inverse).

    Branch k introduces a complementary delay of (depth - 1 - k) * span bits,
    making the total end-to-end delay equal to depth * (depth - 1) * span bits for every branch.

    Parameters
    ----------
    bits : array-like (0s and 1s)
        Interleaved bit stream.
    depth : int
        Number of parallel shift-register branches (default: 8).
    span : int
        Delay increment per branch (default: 4 bits).
    trim_delay : bool
        If True, discards the initial depth * (depth - 1) * span startup latency bits.
    original_len : int | None
        If given, slice output to this length.

    Returns
    -------
    np.ndarray (1D int array)
    """
    bits = np.asarray(bits, dtype=int)
    fifos = [[0] * ((depth - 1 - k) * span) for k in range(depth)]

    out_bits = []
    for i, b in enumerate(bits):
        branch = i % depth
        if fifos[branch]:
            out_bits.append(fifos[branch].pop(0))
            fifos[branch].append(b)
        else:
            out_bits.append(b)

    out_bits = np.array(out_bits, dtype=int)
    total_delay = depth * (depth - 1) * span

    if trim_delay and len(out_bits) >= total_delay:
        out_bits = out_bits[total_delay:]

    if original_len is not None:
        out_bits = out_bits[:original_len]

    return out_bits


def diagonal_interleave(
    bits: np.ndarray | list[int],
    block_size: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """
    Diagonal / pseudo-random permutation-based interleaver.

    Permutes bits within fixed-size blocks according to a reproducible
    pseudo-random permutation generated from `seed`, dispersing burst errors.

    Parameters
    ----------
    bits : array-like (0s and 1s)
        Input bit stream.
    block_size : int
        Size of each permutation block (default: 256).
    seed : int
        PRNG seed for generating reproducible permutation (default: 42).

    Returns
    -------
    np.ndarray (1D int array of length N_blocks * block_size)
    """
    bits = np.asarray(bits, dtype=int)
    n_pad = (block_size - (len(bits) % block_size)) % block_size
    if n_pad > 0:
        padded = np.concatenate([bits, np.zeros(n_pad, dtype=int)])
    else:
        padded = bits

    rng = np.random.RandomState(seed)
    perm = rng.permutation(block_size)

    interleaved = padded.reshape(-1, block_size)[:, perm].reshape(-1)
    return interleaved


def diagonal_deinterleave(
    bits: np.ndarray | list[int],
    block_size: int = 256,
    seed: int = 42,
    original_len: int | None = None,
) -> np.ndarray:
    """
    Inverse diagonal / pseudo-random permutation-based de-interleaver.

    De-interleaves bits back to original sequential order using the inverse
    permutation generated with the same `seed`.

    Parameters
    ----------
    bits : array-like (0s and 1s)
        Interleaved bit stream.
    block_size : int
        Size of each permutation block (default: 256).
    seed : int
        PRNG seed used during interleaving (default: 42).
    original_len : int | None
        If given, slice output to this length to remove trailing pad bits.

    Returns
    -------
    np.ndarray (1D int array)
    """
    bits = np.asarray(bits, dtype=int)
    n_pad = (block_size - (len(bits) % block_size)) % block_size
    if n_pad > 0:
        padded = np.concatenate([bits, np.zeros(n_pad, dtype=int)])
    else:
        padded = bits

    rng = np.random.RandomState(seed)
    perm = rng.permutation(block_size)
    inv_perm = np.argsort(perm)

    deinterleaved = padded.reshape(-1, block_size)[:, inv_perm].reshape(-1)
    if original_len is not None:
        return deinterleaved[:original_len]
    return deinterleaved


# Aliases for explicit scheme naming
pseudo_random_interleave = diagonal_interleave
pseudo_random_deinterleave = diagonal_deinterleave

