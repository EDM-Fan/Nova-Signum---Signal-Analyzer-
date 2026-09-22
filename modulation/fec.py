"""
modulation/fec.py
=================
Forward Error Correction (FEC) module:
1. Convolutional Encoder (K=7, Rate=1/2, G1=171 octal, G2=133 octal).
2. Hard-decision Viterbi Decoder (64-state trellis with traceback).
"""

from __future__ import annotations
import numpy as np

# Standard NASA / CCSDS / GNU Radio K=7 Rate 1/2 polynomials
DEFAULT_G1 = 0o171  # 1111001 (0x79)
DEFAULT_G2 = 0o133  # 1011011 (0x5B)
K = 7
NUM_STATES = 1 << (K - 1)  # 64 states


def _build_trellis(g1: int = DEFAULT_G1, g2: int = DEFAULT_G2):
    """
    Precompute state transition and expected output tables for 64-state trellis.
    Returns:
        next_state : (64, 2) int array -> next state given current state and input bit
        outputs    : (64, 2, 2) int array -> [bit0, bit1] output given state and input
        prev_states: (64, 2) int array -> the 2 previous states that can transition into this state
        prev_inputs: (64, 2) int array -> the input bit corresponding to those previous states
    """
    next_state = np.zeros((NUM_STATES, 2), dtype=int)
    outputs = np.zeros((NUM_STATES, 2, 2), dtype=int)
    prev_states = np.zeros((NUM_STATES, 2), dtype=int)
    prev_inputs = np.zeros((NUM_STATES, 2), dtype=int)
    incoming_count = np.zeros(NUM_STATES, dtype=int)

    for state in range(NUM_STATES):
        for u in (0, 1):
            reg = (u << 6) | state
            b1 = bin(reg & g1).count("1") % 2
            b2 = bin(reg & g2).count("1") % 2
            ns = (u << 5) | (state >> 1)

            next_state[state, u] = ns
            outputs[state, u] = [b1, b2]

            idx = incoming_count[ns]
            prev_states[ns, idx] = state
            prev_inputs[ns, idx] = u
            incoming_count[ns] += 1

    return next_state, outputs, prev_states, prev_inputs


_NEXT_STATE, _OUTPUTS, _PREV_STATES, _PREV_INPUTS = _build_trellis()


def conv_encode(
    bits: np.ndarray | list[int],
    g1: int = DEFAULT_G1,
    g2: int = DEFAULT_G2,
    add_tail: bool = True,
) -> np.ndarray:
    """
    Convolutional encoder (K=7, Rate=1/2).

    Parameters
    ----------
    bits : array-like of int
        Information bits (0s and 1s).
    g1, g2 : int
        Octal generator polynomials (default: 0o171, 0o133).
    add_tail : bool
        If True, appends K-1 = 6 zero flush bits to reset state to 0.

    Returns
    -------
    np.ndarray (1D int array of length 2 * (len(bits) + 6))
    """
    bits = np.asarray(bits, dtype=int)
    if add_tail:
        # 6 zero tail bits to flush shift register back to state 0
        input_bits = np.concatenate([bits, np.zeros(K - 1, dtype=int)])
    else:
        input_bits = bits

    state = 0
    coded = np.empty(2 * len(input_bits), dtype=int)

    for i, u in enumerate(input_bits):
        reg = (int(u) << 6) | state
        coded[2 * i] = bin(reg & g1).count("1") % 2
        coded[2 * i + 1] = bin(reg & g2).count("1") % 2
        state = (int(u) << 5) | (state >> 1)

    return coded


def viterbi_decode(
    coded_bits: np.ndarray | list[int],
    g1: int = DEFAULT_G1,
    g2: int = DEFAULT_G2,
    num_info_bits: int | None = None,
) -> np.ndarray:
    """
    Hard-decision Viterbi Decoder for K=7, Rate 1/2 convolutional code.

    Parameters
    ----------
    coded_bits : array-like of int
        Received coded bits (length must be even).
    g1, g2 : int
        Octal generator polynomials (default: 0o171, 0o133).
    num_info_bits : int | None
        If provided, slices the decoded stream to this exact length
        (stripping tail flush bits and interleaver padding).

    Returns
    -------
    np.ndarray (1D int array of decoded information bits)
    """
    coded = np.asarray(coded_bits, dtype=int)
    n_steps = len(coded) // 2
    if n_steps == 0:
        return np.array([], dtype=int)

    # Path metrics initialized: State 0 starts at 0, all other 63 states at large penalty
    path_metrics = np.full(NUM_STATES, 1e6, dtype=float)
    path_metrics[0] = 0.0

    # Traceback matrix: stores surviving (prev_state, input_bit) for each state at each time step
    survivor_state = np.empty((n_steps, NUM_STATES), dtype=int)
    survivor_input = np.empty((n_steps, NUM_STATES), dtype=int)

    # Forward Viterbi pass
    for t in range(n_steps):
        r0 = coded[2 * t]
        r1 = coded[2 * t + 1]
        new_metrics = np.full(NUM_STATES, 1e6, dtype=float)

        for ns in range(NUM_STATES):
            # Two possible incoming paths into state ns
            s0 = _PREV_STATES[ns, 0]
            u0 = _PREV_INPUTS[ns, 0]
            exp0 = _OUTPUTS[s0, u0]
            bm0 = (r0 != exp0[0]) + (r1 != exp0[1])
            m0 = path_metrics[s0] + bm0

            s1 = _PREV_STATES[ns, 1]
            u1 = _PREV_INPUTS[ns, 1]
            exp1 = _OUTPUTS[s1, u1]
            bm1 = (r0 != exp1[0]) + (r1 != exp1[1])
            m1 = path_metrics[s1] + bm1

            if m0 <= m1:
                new_metrics[ns] = m0
                survivor_state[t, ns] = s0
                survivor_input[t, ns] = u0
            else:
                new_metrics[ns] = m1
                survivor_state[t, ns] = s1
                survivor_input[t, ns] = u1

        path_metrics = new_metrics

    # Traceback pass: start at state with minimum metric (state 0 if flush tail used)
    best_state = 0 if path_metrics[0] < 1e5 else int(np.argmin(path_metrics))
    decoded = np.empty(n_steps, dtype=int)

    curr_state = best_state
    for t in range(n_steps - 1, -1, -1):
        prev_s = survivor_state[t, curr_state]
        in_bit = survivor_input[t, curr_state]
        decoded[t] = in_bit
        curr_state = prev_s

    if num_info_bits is not None:
        return decoded[:num_info_bits]
    return decoded
