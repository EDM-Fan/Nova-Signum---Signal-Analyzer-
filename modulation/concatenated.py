"""
modulation/concatenated.py
==========================
Concatenated Forward Error Correction (FEC) scheme:
- Outer Code: Reed-Solomon RS(64, 48) over GF(2^8) [t=8 byte error correction]
- Inner Code: Convolutional Code (K=7, Rate 1/2, G1=171o, G2=133o) + Viterbi Decoder

CCSDS / NASA Deep-Space Standard Architecture:
The inner Viterbi decoder cleans up high channel bit error rates down to low residual
levels, and the outer Reed-Solomon decoder eliminates any residual byte errors,
producing an error-free waterfall threshold.
"""

from __future__ import annotations
import numpy as np

from modulation.rs_fec import rs_encode, rs_decode
from modulation.fec import conv_encode, viterbi_decode


def concatenated_encode(
    info_data: np.ndarray | bytes | list[int],
    n: int = 64,
    k: int = 48,
) -> np.ndarray:
    """
    Encode an information payload using Concatenated FEC (RS outer + Conv inner).

    Parameters
    ----------
    info_data : np.ndarray | bytes | list[int]
        Either a byte array (length k) or binary bit array (length k * 8).
    n : int
        RS codeword length in bytes (default: 64).
    k : int
        RS message length in bytes (default: 48).

    Returns
    -------
    np.ndarray
        1D int array of inner convolutional coded channel bits (length 2*(n*8 + 6)).
    """
    if isinstance(info_data, (bytes, bytearray)):
        info_bytes = np.frombuffer(info_data, dtype=np.uint8)
    else:
        arr = np.asarray(info_data)
        if arr.dtype == np.uint8 and len(arr) == k:
            info_bytes = arr
        else:
            # Binary bits -> pack to bytes
            bits = arr.astype(np.uint8).ravel()
            if len(bits) != k * 8:
                raise ValueError(f"Expected {k * 8} info bits ({k} bytes), got {len(bits)} bits.")
            info_bytes = np.packbits(bits)

    if len(info_bytes) != k:
        raise ValueError(f"Expected {k} information bytes, got {len(info_bytes)}.")

    # 1. Outer Code: Reed-Solomon encoding (k -> n bytes)
    rs_cw_bytes = rs_encode(info_bytes, n=n, k=k)
    rs_cw_bits = np.unpackbits(rs_cw_bytes)

    # 2. Inner Code: Convolutional encoding (Rate 1/2, K=7)
    coded_bits = conv_encode(rs_cw_bits, add_tail=True)
    return coded_bits


def concatenated_decode(
    received_bits: np.ndarray | list[int],
    n: int = 64,
    k: int = 48,
    return_intermediate: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Decode received channel bits using Concatenated FEC (Viterbi inner -> RS outer).

    Parameters
    ----------
    received_bits : np.ndarray | list[int]
        1D array of binary channel bits.
    n : int
        RS codeword length in bytes (default: 64).
    k : int
        RS message length in bytes (default: 48).
    return_intermediate : bool, optional
        If True, return (decoded_info_bits, viterbi_output_bits).

    Returns
    -------
    np.ndarray (or tuple)
        Decoded information bits (length k * 8).
        Raises ValueError if RS syndrome check cannot correct residual errors.
    """
    rx_bits = np.asarray(received_bits, dtype=np.uint8).ravel()
    min_len = 2 * (n * 8 + 6)
    if len(rx_bits) < min_len:
        raise ValueError(f"Received bitstream too short for concatenated decode (need >= {min_len} bits, got {len(rx_bits)}).")

    # 1. Inner Code: Viterbi decoding
    viterbi_out = viterbi_decode(rx_bits)

    # Extract the n * 8 bits corresponding to the RS codeword
    rs_cw_bits = viterbi_out[: n * 8]
    if len(rs_cw_bits) < n * 8:
        raise ValueError(f"Viterbi output too short for RS({n},{k}) codeword (got {len(rs_cw_bits)} bits, expected {n*8}).")

    rs_cw_bytes = np.packbits(rs_cw_bits)

    # 2. Outer Code: Reed-Solomon decoding (propagates ValueError on uncorrectable error)
    dec_info_bytes = rs_decode(rs_cw_bytes, n=n, k=k)
    dec_info_bits = np.unpackbits(dec_info_bytes)

    if return_intermediate:
        return dec_info_bits, rs_cw_bits
    return dec_info_bits

