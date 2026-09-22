"""
dsp/rate_estimator.py
=====================
Automated sampling-frequency and symbol-rate estimator for raw IQ signals.

Features:
1. Cyclostationary envelope spectrum peak detection (for PSK/QAM with pulse shaping).
2. Instantaneous frequency deviation & transition detection fallback (for continuous-phase 2FSK).
3. Candidate rate scoring table comparing candidate sample rates against integer SPS and canonical rates.
"""

from __future__ import annotations
import numpy as np

# Standard canonical symbol rates (in baud / sym/s)
CANONICAL_SYMBOL_RATES = np.array([
    125_000, 100_000, 50_000, 25_000, 19_200, 9_600, 4_800, 2_400, 1_200
])

# Standard FSK tone separation values (2 * deviation, in Hz)
CANONICAL_FSK_SEPARATIONS = np.array([
    50_000, 25_000, 10_000, 5_000, 2_400
])

DEFAULT_CANDIDATES = [1_000_000, 2_000_000, 250_000, 48_000]


def _score_rate_match(val: float, target_list: np.ndarray, tolerance: float = 0.20) -> float:
    """Score proximity of a continuous value to a list of target reference values."""
    rel_diffs = np.abs(target_list - val) / (target_list + 1e-12)
    min_diff = np.min(rel_diffs)
    if min_diff < tolerance:
        return float(1.0 - (min_diff / tolerance))
    return 0.0


def _estimate_fsk_rate(samples: np.ndarray, candidate_rates: list[int]) -> dict:
    """
    FSK fallback: Detect tone separation and symbol rate from instantaneous frequency transitions.
    """
    import scipy.signal as sig

    phase = np.unwrap(np.angle(samples))
    inst_freq = np.diff(phase) / (2.0 * np.pi)  # in cycles/sample

    # Filter out noisy transitions: use median of positive & negative deviations
    pos_samples = inst_freq[inst_freq > 0.005]
    neg_samples = inst_freq[inst_freq < -0.005]
    pos_tone = float(np.median(pos_samples)) if len(pos_samples) > 10 else 0.025
    neg_tone = float(np.median(neg_samples)) if len(neg_samples) > 10 else -0.025
    tone_sep_norm = abs(pos_tone - neg_tone)

    # Median filter to eliminate sample-level differentiator noise
    med_freq = sig.medfilt(inst_freq, kernel_size=5)
    med_zero = med_freq - np.median(med_freq)

    # Detect zero crossings / symbol level transitions
    zero_cross = np.where(np.diff(np.sign(med_zero)) != 0)[0]
    intervals = np.diff(zero_cross)
    valid_intervals = intervals[intervals >= 2]

    if len(valid_intervals) > 0:
        counts = np.bincount(valid_intervals)
        mode_sps = int(np.argmax(counts))
        # Cluster around mode to get precise median SPS
        cluster = valid_intervals[np.abs(valid_intervals - mode_sps) <= max(1, mode_sps // 4)]
        est_sps = float(np.median(cluster)) if len(cluster) else float(mode_sps)
    else:
        est_sps = 8.0

    est_sps = max(2.0, min(64.0, est_sps))
    f_sym_norm = 1.0 / est_sps

    candidate_scores = {}
    for fs in candidate_rates:
        tone_sep_hz = fs * tone_sep_norm
        sym_rate_hz = fs * f_sym_norm

        sep_score = _score_rate_match(tone_sep_hz, CANONICAL_FSK_SEPARATIONS, tolerance=0.20)
        sym_score = _score_rate_match(sym_rate_hz, CANONICAL_SYMBOL_RATES, tolerance=0.20)
        sps_val = fs / (sym_rate_hz + 1e-12)
        int_sps_score = max(0.0, 1.0 - abs(sps_val - round(sps_val)) * 3.0)

        # Joint scoring: tone separation + symbol rate + integer SPS
        total_score = 0.50 * sep_score + 0.35 * sym_score + 0.15 * int_sps_score
        candidate_scores[fs] = float(np.clip(total_score, 0.05, 0.98))

    best_rate = max(candidate_scores, key=candidate_scores.get)
    confidence = candidate_scores[best_rate] * 100.0

    return {
        "best_rate": best_rate,
        "confidence": confidence,
        "estimated_sps": est_sps,
        "estimated_sym_rate": float(best_rate * f_sym_norm),
        "method": "fsk_frequency_deviation",
        "candidate_scores": candidate_scores,
    }


def estimate_sample_rate(
    samples: np.ndarray,
    candidate_rates: list[int] | None = None,
) -> dict:
    """
    Estimate the true sampling frequency from raw IQ samples by evaluating
    cyclostationary symbol clock peaks against candidate rates.

    Parameters
    ----------
    samples : np.ndarray (complex)
        Raw IQ samples.
    candidate_rates : list[int] | None
        List of candidate sample rates to rank (default: [1M, 2M, 250k, 48k]).

    Returns
    -------
    dict with keys:
        best_rate          : int    Winning candidate rate in Hz
        confidence         : float  Confidence score (0 - 100%)
        estimated_sps      : float  Detected samples-per-symbol
        estimated_sym_rate : float  Estimated symbol rate at winning sample rate (sym/s)
        method             : str    'envelope_cyclostationary' or 'fsk_frequency_deviation'
        candidate_scores   : dict   Full candidate rate -> score mapping [0.0, 1.0]
    """
    if candidate_rates is None:
        candidate_rates = DEFAULT_CANDIDATES

    samples = np.asarray(samples, dtype=np.complex128)

    # Note: rate estimation uses only the first 32,768 samples — if the signal of interest
    # doesn't start near the beginning of the file (e.g. leading silence/noise), the estimate
    # may be less reliable. Acceptable limitation for this demo scope.
    MAX_SAMPLES = 32_768
    if len(samples) > MAX_SAMPLES:
        samples = samples[:MAX_SAMPLES]

    n = len(samples)
    if n < 256:
        raise ValueError("Sample array too short for rate estimation.")

    # 1. Envelope cyclostationary non-linearity: y[n] = |x[n]|^2
    amp = np.abs(samples)
    amp_var = np.var(amp) / (np.mean(amp) ** 2 + 1e-12)

    # If envelope is flat (< 0.03 normalized variance), trigger FSK fallback
    if amp_var < 0.03:
        return _estimate_fsk_rate(samples, candidate_rates)

    y = amp ** 2 - np.mean(amp ** 2)
    win = np.hanning(n)
    Y = np.abs(np.fft.fft(y * win))[: n // 2]
    freqs = np.fft.fftfreq(n, d=1.0)[: n // 2]

    # Search for symbol rate peak in normalized frequency range [0.01, 0.49]
    search_mask = (freqs >= 0.01) & (freqs <= 0.49)
    if not np.any(search_mask):
        return _estimate_fsk_rate(samples, candidate_rates)

    search_Y = Y[search_mask]
    search_freqs = freqs[search_mask]

    peak_idx = np.argmax(search_Y)
    f_sym_norm = float(search_freqs[peak_idx])
    p_peak = float(search_Y[peak_idx])
    p_median = float(np.median(search_Y)) + 1e-12

    prominence_db = 10.0 * np.log10(p_peak / p_median)

    # If cyclostationary peak prominence is very weak (< 4.5 dB), fallback to FSK
    if prominence_db < 4.5:
        return _estimate_fsk_rate(samples, candidate_rates)

    est_sps = 1.0 / f_sym_norm

    # 2. Score each candidate rate
    candidate_scores = {}
    for fs in candidate_rates:
        est_baud = fs * f_sym_norm
        baud_match = _score_rate_match(est_baud, CANONICAL_SYMBOL_RATES, tolerance=0.10)
        sps_val = fs / (est_baud + 1e-12)
        int_sps_match = max(0.0, 1.0 - abs(sps_val - round(sps_val)) * 3.0)

        # Base confidence from prominence
        prom_score = np.clip((prominence_db - 4.0) / 16.0, 0.1, 1.0)

        score = prom_score * (0.65 * baud_match + 0.35 * int_sps_match)
        candidate_scores[fs] = float(np.clip(score, 0.05, 0.98))

    best_rate = max(candidate_scores, key=candidate_scores.get)
    confidence = candidate_scores[best_rate] * 100.0

    return {
        "best_rate": best_rate,
        "confidence": confidence,
        "estimated_sps": float(est_sps),
        "estimated_sym_rate": float(best_rate * f_sym_norm),
        "method": "envelope_cyclostationary",
        "candidate_scores": candidate_scores,
    }
