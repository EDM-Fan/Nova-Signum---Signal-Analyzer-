"""
modulation/correlator.py
========================
Bit-stream correlation for preamble / sync word detection and
automatic carrier phase ambiguity resolution (BPSK inversion, QPSK quadrant rotation).
"""

from __future__ import annotations
import numpy as np

# 32-bit standard frame sync word (0xEB902A3C in hex)
DEFAULT_SYNC_32 = np.array([
    1, 1, 1, 0, 1, 0, 1, 1,  # 0xEB
    1, 0, 0, 1, 0, 0, 0, 0,  # 0x90
    0, 0, 1, 0, 1, 0, 1, 0,  # 0x2A
    0, 0, 1, 1, 1, 1, 0, 0,  # 0x3C
], dtype=int)

# CCSDS standard 32-bit sync word / Alt A (0x1ACFFC1D)
SYNC_CCSDS_32 = np.array([
    0, 0, 0, 1, 1, 0, 1, 0,  # 0x1A
    1, 1, 0, 0, 1, 1, 1, 1,  # 0xCF
    1, 1, 1, 1, 1, 1, 0, 0,  # 0xFC
    0, 0, 0, 1, 1, 1, 0, 1,  # 0x1D
], dtype=int)

# Alternate sync pattern B (0xFAF334BE)
SYNC_ALT_B_32 = np.array([
    1, 1, 1, 1, 1, 0, 1, 0,  # 0xFA
    1, 1, 1, 1, 0, 0, 1, 1,  # 0xF3
    0, 0, 1, 1, 0, 1, 0, 0,  # 0x34
    1, 0, 1, 1, 1, 1, 1, 0,  # 0xBE
], dtype=int)

# Alternate sync pattern C (0x352EF853)
SYNC_ALT_C_32 = np.array([
    0, 0, 1, 1, 0, 1, 0, 1,  # 0x35
    0, 0, 1, 0, 1, 1, 1, 0,  # 0x2E
    1, 1, 1, 1, 1, 0, 0, 0,  # 0xF8
    0, 1, 0, 1, 0, 0, 1, 1,  # 0x53
], dtype=int)

SYNC_PRESETS: dict[str, np.ndarray] = {
    "default": DEFAULT_SYNC_32,
    "0xEB902A3C": DEFAULT_SYNC_32,
    "Default (0xEB902A3C)": DEFAULT_SYNC_32,
    "ccsds": SYNC_CCSDS_32,
    "0x1ACFFC1D": SYNC_CCSDS_32,
    "Alt A (0x1ACFFC1D)": SYNC_CCSDS_32,
    "alt_b": SYNC_ALT_B_32,
    "0xFAF334BE": SYNC_ALT_B_32,
    "Alt B (0xFAF334BE)": SYNC_ALT_B_32,
    "alt_c": SYNC_ALT_C_32,
    "0x352EF853": SYNC_ALT_C_32,
    "Alt C (0x352EF853)": SYNC_ALT_C_32,
}


def hex_to_sync_bits(hex_val: str, bit_length: int = 32) -> np.ndarray:
    """Convert a hex string (e.g. '0xEB902A3C' or 'EB902A3C') to a bit array."""
    s = hex_val.strip()
    if s.lower().startswith("0x"):
        s = s[2:]
    val = int(s, 16)
    bits = [(val >> (bit_length - 1 - i)) & 1 for i in range(bit_length)]
    return np.array(bits, dtype=int)


def parse_sync_word(sync_input: np.ndarray | list[int] | str | None) -> np.ndarray:
    """
    Parse a sync pattern from an array, list, preset name, hex string, or binary string.
    """
    if sync_input is None:
        return DEFAULT_SYNC_32

    if isinstance(sync_input, (np.ndarray, list)):
        return np.asarray(sync_input, dtype=int)

    if isinstance(sync_input, str):
        s = sync_input.strip()
        if s in SYNC_PRESETS:
            return SYNC_PRESETS[s]
        # Check if formatted like "Alt A (0x1ACFFC1D)"
        if "(" in s and ")" in s:
            inner = s[s.find("(") + 1 : s.find(")")].strip()
            if inner in SYNC_PRESETS:
                return SYNC_PRESETS[inner]
            if inner.lower().startswith("0x") or all(c in "0123456789abcdefABCDEF" for c in inner):
                return hex_to_sync_bits(inner)

        # Check if pure binary string (e.g. "11101011...")
        if all(c in "01" for c in s) and len(s) >= 8:
            return np.array([int(c) for c in s], dtype=int)

        # Check if hex string
        clean_hex = s[2:] if s.lower().startswith("0x") else s
        if all(c in "0123456789abcdefABCDEF" for c in clean_hex) and len(clean_hex) > 0:
            bit_len = len(clean_hex) * 4
            return hex_to_sync_bits(clean_hex, bit_length=bit_len)

        raise ValueError(f"Unrecognized sync word format: '{sync_input}'")

    raise TypeError(f"Unsupported sync_input type: {type(sync_input)}")


def _generate_candidate_bitstreams(bits: np.ndarray, modulation_type: str) -> dict[str, np.ndarray]:
    """Generate phase ambiguity candidate bitstreams based on modulation type."""
    mod = modulation_type.upper().strip()
    candidates = {}

    if mod == "BPSK":
        candidates["direct (0 deg)"] = bits
        candidates["inverted (180 deg)"] = 1 - bits

    elif mod in ("2FSK", "2-FSK", "FSK"):
        candidates["direct (0 deg)"] = bits
        candidates["inverted (180 deg)"] = 1 - bits

    elif mod == "QPSK":
        # Ensure even length for I/Q de-interleaving
        n_pairs = len(bits) // 2
        i_ch = bits[: 2 * n_pairs : 2]
        q_ch = bits[1 : 2 * n_pairs : 2]

        # 4 quadrant rotational symmetries:
        # 0 deg:   ( I,  Q) -> I, Q
        # 90 deg:  (-Q,  I) -> (1-Q), I
        # 180 deg: (-I, -Q) -> (1-I), (1-Q)
        # 270 deg: ( Q, -I) -> Q, (1-I)
        rotations = {
            "0 deg": (i_ch, q_ch),
            "90 deg": (1 - q_ch, i_ch),
            "180 deg": (1 - i_ch, 1 - q_ch),
            "270 deg": (q_ch, 1 - i_ch),
        }

        for name, (rot_i, rot_q) in rotations.items():
            cand = np.empty(2 * n_pairs, dtype=int)
            cand[0::2] = rot_i
            cand[1::2] = rot_q
            candidates[name] = cand

    elif mod in ("16QAM", "16-QAM"):
        # 4 bits per symbol: [b_I0, b_Q0, b_I1, b_Q1]
        n_quads = len(bits) // 4
        b_i0 = bits[0 : 4 * n_quads : 4]
        b_q0 = bits[1 : 4 * n_quads : 4]
        b_i1 = bits[2 : 4 * n_quads : 4]
        b_q1 = bits[3 : 4 * n_quads : 4]

        # 4 quadrant rotational symmetries under Gray code:
        # 0 deg:   ( I0,  Q0, I1, Q1) -> (b_i0, b_q0, b_i1, b_q1)
        # 90 deg:  (-Q0,  I0, Q1, I1) -> (1-b_q0, b_i0, b_q1, b_i1)
        # 180 deg: (-I0, -Q0, I1, Q1) -> (1-b_i0, 1-b_q0, b_i1, b_q1)
        # 270 deg: ( Q0, -I0, Q1, I1) -> (b_q0, 1-b_i0, b_q1, b_i1)
        rotations_16qam = {
            "0 deg":   (b_i0, b_q0, b_i1, b_q1),
            "90 deg":  (1 - b_q0, b_i0, b_q1, b_i1),
            "180 deg": (1 - b_i0, 1 - b_q0, b_i1, b_q1),
            "270 deg": (b_q0, 1 - b_i0, b_q1, b_i1),
        }

        for name, (r_i0, r_q0, r_i1, r_q1) in rotations_16qam.items():
            cand = np.empty(4 * n_quads, dtype=int)
            cand[0::4] = r_i0
            cand[1::4] = r_q0
            cand[2::4] = r_i1
            cand[3::4] = r_q1
            candidates[name] = cand

    else:
        # Default fallback
        candidates["direct"] = bits

    return candidates


def find_sync(
    bits: np.ndarray,
    sync_pattern: np.ndarray | list[int] | str | None = None,
    modulation_type: str = "BPSK",
) -> dict:
    """
    Search for a known sync pattern in a demodulated bit stream across all
    phase ambiguity variants (BPSK inversion / QPSK rotations).

    Parameters
    ----------
    bits : np.ndarray
        1D array of demodulated bits (0s and 1s).
    sync_pattern : np.ndarray | list[int] | str | None
        Known sync bit sequence, preset name, or hex string. If None, uses DEFAULT_SYNC_32.
    modulation_type : str
        'BPSK' or 'QPSK'.

    Returns
    -------
    dict with keys:
        offset         : int         Bit index where sync pattern begins
        score          : float       Normalized correlation peak [-1.0, 1.0] (1.0 = 100% match)
        variant        : str         Phase ambiguity variant that yielded the peak
        corrected_bits : np.ndarray  The full bitstream rotated by the detected variant
        payload_bits   : np.ndarray  Bits immediately following the sync pattern
    """
    bits = np.asarray(bits, dtype=int)
    sync_pattern = parse_sync_word(sync_pattern)

    L = len(sync_pattern)
    if len(bits) < L:
        raise ValueError(f"Bit stream length ({len(bits)}) is shorter than sync pattern ({L}).")

    sync_bipolar = 2 * sync_pattern - 1

    candidates = _generate_candidate_bitstreams(bits, modulation_type)

    best_score = -2.0
    best_offset = -1
    best_variant = ""
    best_corrected_bits = bits

    for variant_name, cand_bits in candidates.items():
        if len(cand_bits) < L:
            continue
        cand_bipolar = 2 * cand_bits - 1
        # Valid cross-correlation: sliding window of length L
        corr = np.correlate(cand_bipolar, sync_bipolar, mode="valid") / float(L)

        peak_idx = int(np.argmax(corr))
        peak_val = float(corr[peak_idx])

        if peak_val > best_score:
            best_score = peak_val
            best_offset = peak_idx
            best_variant = variant_name
            best_corrected_bits = cand_bits

    payload_start = best_offset + L
    payload_bits = best_corrected_bits[payload_start:]

    return {
        "offset": best_offset,
        "score": best_score,
        "score_warn": best_score < 0.75,  # True → low-confidence lock, may be noise
        "variant": best_variant,
        "corrected_bits": best_corrected_bits,
        "payload_bits": payload_bits,
    }
