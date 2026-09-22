"""
modulation/demodulator.py
=========================
Digital demodulation for BPSK, QPSK, 16QAM, and 2FSK complex baseband signals.

Performs:
1. BPSK / QPSK / 16QAM:
   - RRC matched filtering
   - M-th power carrier frequency offset (CFO) estimation & compensation
   - Maximum-power symbol timing recovery & decimation
   - Constellation power normalization (16QAM)
   - Carrier phase offset (CPO) alignment
   - Gray-coded symbol slicing to recovered bit array
2. 2FSK:
   - Instantaneous frequency discriminator (differentiator via conjugate delay product)
   - Center-frequency / CFO removal via median tone centering
   - SPS-matched moving average integration filter
   - Symbol clock timing recovery via maximum-variance phase decimation
   - Frequency threshold slicing (+df -> 1, -df -> 0)
"""

from __future__ import annotations
import numpy as np

try:
    from modulation.signal_gen import _rrc_filter
except ImportError:
    from signal_gen import _rrc_filter


def _estimate_cfo(samples: np.ndarray, sample_rate: int, M: int) -> float:
    """Estimate carrier frequency offset using M-th power non-linear method."""
    z = samples ** M
    n = len(z)
    win = np.hanning(n)
    Z = np.fft.fft(z * win)
    freqs = np.fft.fftfreq(n, d=1.0 / sample_rate)
    peak_idx = np.argmax(np.abs(Z))
    return float(freqs[peak_idx] / M)


def demodulate(
    samples: np.ndarray,
    sample_rate: int,
    modulation_type: str,
    sps: int = 8,
    return_symbols: bool = False,
    deviation_hz: float = 25_000.0,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Demodulate BPSK, QPSK, 16QAM, or 2FSK IQ samples into recovered bits.

    Parameters
    ----------
    samples : np.ndarray (complex)
        Raw baseband IQ samples.
    sample_rate : int
        Sample rate in Hz.
    modulation_type : str
        Predicted modulation: 'BPSK', 'QPSK', '16QAM', or '2FSK'.
    sps : int
        Samples per symbol (default: 8).
    return_symbols : bool, optional
        If True, return a tuple (bits, symbols) where symbols is the complex
        recovered symbol array. Default is False.
    deviation_hz : float, optional
        Frequency deviation for 2FSK in Hz (default: 25,000 Hz).

    Returns
    -------
    np.ndarray (1D int array of 0s and 1s) or tuple of (bits, symbols)
    """
    mod = modulation_type.upper().strip()
    if mod not in ("BPSK", "QPSK", "16QAM", "16-QAM", "2FSK", "2-FSK", "FSK"):
        raise ValueError(
            f"Demodulation for '{modulation_type}' is not supported. "
            "Supported: BPSK, QPSK, 16QAM, 2FSK."
        )

    samples = np.asarray(samples, dtype=np.complex128).ravel()
    if len(samples) < sps * 8:
        raise ValueError("Signal too short for demodulation.")

    # =========================================================================
    # 1. 2FSK Demodulation (Frequency Discriminator Pipeline)
    # =========================================================================
    if mod in ("2FSK", "2-FSK", "FSK"):
        # a. Instantaneous frequency via conjugate delay product
        phase_diff = np.angle(samples[1:] * np.conj(samples[:-1]))
        inst_freq = phase_diff * (sample_rate / (2.0 * np.pi))

        # b. CFO estimation & tone centering
        cfo = float(np.median(inst_freq))
        base_freq = inst_freq - cfo

        # c. SPS-length moving average integration filter
        filt = np.ones(sps, dtype=float) / float(sps)
        smoothed_freq = np.convolve(base_freq, filt, mode="same")

        # d. Symbol clock recovery: phase maximizing symbol eye variance
        best_phase = int(np.argmax([np.var(smoothed_freq[p::sps]) for p in range(sps)]))
        sym_freqs = smoothed_freq[best_phase::sps]

        # e. Frequency threshold slicing (+df -> 1, -df -> 0)
        bits = (sym_freqs > 0).astype(int)

        # f. Normalized pseudo-symbols for constellation / scope display
        norm_scale = deviation_hz if deviation_hz > 0 else (np.mean(np.abs(sym_freqs)) + 1e-12)
        symbols = (sym_freqs / norm_scale) + 0j

        if return_symbols:
            return bits, symbols
        return bits

    # =========================================================================
    # 2. Linear PSK / QAM Demodulation (Matched Filter Pipeline)
    # =========================================================================
    # Matched filtering using exact RRC filter parameters from signal_gen
    h = _rrc_filter(beta=0.35, span=8, sps=sps)
    mf = np.convolve(samples, h, mode="same")

    # CFO estimation & compensation (M=2 for BPSK, M=4 for QPSK/16QAM)
    M = 2 if mod == "BPSK" else 4
    cfo = _estimate_cfo(mf, sample_rate, M)
    t = np.arange(len(mf)) / sample_rate
    mf = mf * np.exp(-1j * 2.0 * np.pi * cfo * t)

    # Symbol Timing Recovery (sampling phase with maximum symbol power)
    best_phase = int(np.argmax([np.mean(np.abs(mf[phase::sps]) ** 2) for phase in range(sps)]))
    symbols = mf[best_phase::sps]

    # Carrier Phase Offset (CPO) de-rotation & slicing
    if mod == "BPSK":
        # 2nd-harmonic phase estimate
        phase_err = 0.5 * np.angle(np.sum(symbols ** 2))
        symbols = symbols * np.exp(-1j * phase_err)

        # Slicing: +1 -> 1, -1 -> 0
        bits = (np.real(symbols) > 0).astype(int)

    elif mod == "QPSK":
        # 4th-power phase estimate: (-1) * sum(s^4) has angle 4*theta
        phase_err = 0.25 * np.angle(-np.sum(symbols ** 4))
        symbols = symbols * np.exp(-1j * phase_err)

        # Slicing & bit interleaving: [I0, Q0, I1, Q1, ...]
        b_i = (np.real(symbols) > 0).astype(int)
        b_q = (np.imag(symbols) > 0).astype(int)

        bits = np.empty(2 * len(symbols), dtype=int)
        bits[0::2] = b_i
        bits[1::2] = b_q

    elif mod in ("16QAM", "16-QAM"):
        # Power normalization (target E[|s|^2] = 1.0)
        avg_power = np.mean(np.abs(symbols) ** 2)
        if avg_power > 1e-12:
            symbols = symbols / np.sqrt(avg_power)

        # 4th-power phase alignment (aligned 16QAM constellation has E[s^4] < 0)
        phase_err = 0.25 * np.angle(-np.sum(symbols ** 4))
        symbols = symbols * np.exp(-1j * phase_err)

        # Scale to canonical integer grid levels {-3, -1, +1, +3}
        scaled = symbols * np.sqrt(10.0)
        i_val = np.real(scaled)
        q_val = np.imag(scaled)

        # Gray decoding for 4 PAM levels {-3, -1, +1, +3}:
        # b0 (sign):      1 if level > 0 else 0
        # b1 (magnitude): 1 if |level| < 2.0 else 0
        b_i0 = (i_val > 0).astype(int)
        b_q0 = (q_val > 0).astype(int)
        b_i1 = (np.abs(i_val) < 2.0).astype(int)
        b_q1 = (np.abs(q_val) < 2.0).astype(int)

        # Interleave 4 bits per symbol: [b_I0, b_Q0, b_I1, b_Q1]
        n_sym = len(symbols)
        bits = np.empty(4 * n_sym, dtype=int)
        bits[0::4] = b_i0
        bits[1::4] = b_q0
        bits[2::4] = b_i1
        bits[3::4] = b_q1

    if return_symbols:
        return bits, symbols
    return bits
