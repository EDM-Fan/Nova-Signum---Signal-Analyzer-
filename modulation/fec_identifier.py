"""
modulation/fec_identifier.py
============================
Blind FEC & Interleaving Scheme Identification module.

Uses a trial-and-consistency approach across all supported FEC / interleaving schemes:
1. Viterbi + Block (8x32) Interleaver (Rate-1/2, K=7)
2. Viterbi + Convolutional (8x4) Interleaver (Rate-1/2, K=7)
3. Viterbi + Pseudo-Random Diagonal (256-bit seeded permutation) Interleaver
4. Viterbi + True Diagonal (16x16 deterministic cyclic-shift) Interleaver
5. Reed-Solomon RS(32, 24)  [GF(2^8), t=4]
6. Reed-Solomon RS(64, 48)  [GF(2^8), t=8]
7. Reed-Solomon RS(128, 112) [GF(2^8), t=8]
8. LDPC (128, 64) Rate-1/2
9. Concatenated RS(64,48) + Viterbi

Confidence scoring principles:
- Reed-Solomon: Strict BCH/Chien algebraic syndrome check + re-encode consistency.
- LDPC: Parity check syndrome check H @ c^T == 0 (mod 2) on Min-Sum convergence.
- Viterbi + Interleaver: Re-encode decoded payload and compute normalized Hamming distance
  against received channel bits. True interleaver yields distance near channel BER (< 5%);
  mismatched interleavers yield near-random distances (~ 20-50%).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from modulation.fec import conv_encode, viterbi_decode
from modulation.deinterleaver import (
    block_interleave,
    block_deinterleave,
    conv_interleave,
    conv_deinterleave,
    diagonal_interleave,
    diagonal_deinterleave,
    true_diagonal_interleave,
    true_diagonal_deinterleave,
)
from modulation.rs_fec import rs_encode, rs_decode
from modulation.ldpc import ldpc_encode, ldpc_decode
from modulation.concatenated import concatenated_encode, concatenated_decode


@dataclass
class SchemeResult:
    name: str
    scheme_key: str
    rs_size: tuple[int, int] | None
    confidence: float
    decoded_bits: np.ndarray
    details: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "scheme_key": self.scheme_key,
            "rs_size": self.rs_size,
            "confidence": self.confidence,
            "decoded_bits": self.decoded_bits,
            "details": self.details,
        }


# =============================================================================
# Statistical Noise-Floor Distribution Parameters (Empirically Characterized)
# =============================================================================
# Measured over 3,000 independent trials of uniform random noise per scheme:
#
#   Scheme                       Mean (mu)   Std (sigma)   Threshold (mu - 3*sigma)   Empirical FA
#   ------------------------------------------------------------------------------------------------
#   Viterbi + Block                0.14938     0.00795             0.1255                  0.13%
#   Viterbi + Conv                 0.30583     0.01579             0.2585                  0.07%
#   Viterbi + Pseudo-Rand Diag     0.14931     0.00799             0.1253                  0.10%
#   Viterbi + True Diagonal        0.14931     0.00799             0.1253                  0.10%
#     (same as pseudo-rand: both use 256-bit blocks & identical Viterbi; the
#      bijective permutation does not change the noise-floor d_H distribution)
# =============================================================================

_VITERBI_NOISE_PROFILES = {
    "viterbi_block":         {"mu": 0.14938, "sigma": 0.00795, "k": 3.0},
    "viterbi_conv":          {"mu": 0.30583, "sigma": 0.01579, "k": 3.0},
    "viterbi_diagonal":      {"mu": 0.14931, "sigma": 0.00799, "k": 3.0},
    "viterbi_true_diagonal": {"mu": 0.14931, "sigma": 0.00799, "k": 3.0},
}

# Minimum confidence floor for identify_fec_scheme to declare a winner.
# If no candidate clears this floor, the identifier returns [] (unidentified).
_MIN_CONFIDENCE = 0.20


def _viterbi_confidence(d: float, scheme_key: str) -> float:
    """
    Compute confidence for a Viterbi interleaver scheme based on its empirical
    noise-floor z-score distribution.

    A scheme receives non-zero confidence only if its re-encoding Hamming distance
    d is at least k standard deviations below its empirical noise-floor mean:
        d_thresh = mu - k * sigma
    """
    profile = _VITERBI_NOISE_PROFILES[scheme_key]
    mu, sigma, k = profile["mu"], profile["sigma"], profile["k"]
    thresh = mu - k * sigma
    if d >= thresh:
        return 0.0
    return float(np.clip(1.0 - (d / thresh) ** 1.5, 0.0, 1.0))


def identify_fec_scheme(payload_bits: np.ndarray | list[int]) -> list[SchemeResult]:
    """
    Blindly evaluate candidate FEC & interleaving schemes on the received payload.

    Parameters:
        payload_bits: 1D array of binary bits {0, 1} extracted after sync detection.

    Returns:
        List of SchemeResult objects ranked by confidence in descending order.
    """
    payload = np.asarray(payload_bits, dtype=np.uint8).ravel()
    L = len(payload)
    results: list[SchemeResult] = []

    if L == 0:
        return results

    # -------------------------------------------------------------------------
    # 1. Reed-Solomon RS(32, 24)
    # -------------------------------------------------------------------------
    if L >= 256:
        try:
            cw_bytes = np.packbits(payload[:256])
            dec_bytes = rs_decode(cw_bytes, n=32, k=24)
            re_cw = rs_encode(dec_bytes, n=32, k=24)
            byte_errs = int(np.sum(cw_bytes != re_cw))
            if byte_errs <= 4:
                conf = float(1.0 - (byte_errs / 32.0) * 0.4)
                dec_bits = np.unpackbits(dec_bytes)
                results.append(
                    SchemeResult(
                        name="Reed-Solomon RS(32,24)",
                        scheme_key="rs",
                        rs_size=(32, 24),
                        confidence=conf,
                        decoded_bits=dec_bits,
                        details=f"RS(32,24) syndrome valid ({byte_errs}/4 byte errors corrected)",
                    )
                )
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 2. Reed-Solomon RS(64, 48)
    # -------------------------------------------------------------------------
    if L >= 512:
        try:
            cw_bytes = np.packbits(payload[:512])
            dec_bytes = rs_decode(cw_bytes, n=64, k=48)
            re_cw = rs_encode(dec_bytes, n=64, k=48)
            byte_errs = int(np.sum(cw_bytes != re_cw))
            if byte_errs <= 8:
                conf = float(1.0 - (byte_errs / 64.0) * 0.4)
                dec_bits = np.unpackbits(dec_bytes)
                results.append(
                    SchemeResult(
                        name="Reed-Solomon RS(64,48)",
                        scheme_key="rs",
                        rs_size=(64, 48),
                        confidence=conf,
                        decoded_bits=dec_bits,
                        details=f"RS(64,48) syndrome valid ({byte_errs}/8 byte errors corrected)",
                    )
                )
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 3. Reed-Solomon RS(128, 112)
    # -------------------------------------------------------------------------
    if L >= 1024:
        try:
            cw_bytes = np.packbits(payload[:1024])
            dec_bytes = rs_decode(cw_bytes, n=128, k=112)
            re_cw = rs_encode(dec_bytes, n=128, k=112)
            byte_errs = int(np.sum(cw_bytes != re_cw))
            if byte_errs <= 8:
                conf = float(1.0 - (byte_errs / 128.0) * 0.4)
                dec_bits = np.unpackbits(dec_bytes)
                results.append(
                    SchemeResult(
                        name="Reed-Solomon RS(128,112)",
                        scheme_key="rs",
                        rs_size=(128, 112),
                        confidence=conf,
                        decoded_bits=dec_bits,
                        details=f"RS(128,112) syndrome valid ({byte_errs}/8 byte errors corrected)",
                    )
                )
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 4. LDPC (128, 64)
    # -------------------------------------------------------------------------
    if L >= 128:
        try:
            cw_bits = payload[:128]
            dec_info, conv, iters = ldpc_decode(cw_bits, max_iterations=50)
            if conv:
                re_cw = ldpc_encode(dec_info)
                bit_errs = int(np.sum(cw_bits != re_cw))
                ber = bit_errs / 128.0
                conf = float(np.clip(1.0 - ber * 0.5 - 0.05 * (iters / 50.0), 0.0, 1.0))
                results.append(
                    SchemeResult(
                        name="LDPC (128,64)",
                        scheme_key="ldpc",
                        rs_size=None,
                        confidence=conf,
                        decoded_bits=dec_info,
                        details=f"Parity checks satisfied in {iters} iterations ({bit_errs} bit errors corrected)",
                    )
                )
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 5. Viterbi + Block Interleaver (8x32)
    # -------------------------------------------------------------------------
    if L >= 256:
        try:
            deint = block_deinterleave(payload, rows=8, cols=32)
            u_hat = viterbi_decode(deint)
            c_re = conv_encode(u_hat, add_tail=True)
            re_int = block_interleave(c_re, rows=8, cols=32)
            n_chk = min(L, len(re_int))
            d = float(np.mean(payload[:n_chk] != re_int[:n_chk]))
            conf = _viterbi_confidence(d, "viterbi_block")
            if conf > 0.0:
                pad_note = " (matched with minor block padding overhead)" if 0.015 <= d <= 0.05 else ""
                results.append(
                    SchemeResult(
                        name="Viterbi + Block (8x32)",
                        scheme_key="viterbi_block",
                        rs_size=None,
                        confidence=conf,
                        decoded_bits=u_hat,
                        details=f"Re-encode consistency match: {100.0 * (1.0 - d):.1f}% (d_H={d*100:.1f}%){pad_note}",
                    )
                )
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 6. Viterbi + Convolutional Interleaver (8x4)
    # -------------------------------------------------------------------------
    if L >= 256:
        try:
            deint = conv_deinterleave(payload, depth=8, span=4, trim_delay=True)
            u_hat = viterbi_decode(deint)
            c_re = conv_encode(u_hat, add_tail=True)
            re_int = conv_interleave(c_re, depth=8, span=4, flush=True)
            n_chk = min(L, len(re_int))
            d = float(np.mean(payload[:n_chk] != re_int[:n_chk]))
            conf = _viterbi_confidence(d, "viterbi_conv")
            if conf > 0.0:
                results.append(
                    SchemeResult(
                        name="Viterbi + Conv (8x4)",
                        scheme_key="viterbi_conv",
                        rs_size=None,
                        confidence=conf,
                        decoded_bits=u_hat,
                        details=f"Re-encode consistency match: {100.0 * (1.0 - d):.1f}% (d_H={d*100:.1f}%)",
                    )
                )
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 7. Viterbi + Pseudo-Random Diagonal Interleaver (256-bit seeded)
    # -------------------------------------------------------------------------
    if L >= 256:
        try:
            deint = diagonal_deinterleave(payload, block_size=256, seed=42)
            u_hat = viterbi_decode(deint)
            c_re = conv_encode(u_hat, add_tail=True)
            re_int = diagonal_interleave(c_re, block_size=256, seed=42)
            n_chk = min(L, len(re_int))
            d = float(np.mean(payload[:n_chk] != re_int[:n_chk]))
            conf = _viterbi_confidence(d, "viterbi_diagonal")
            if conf > 0.0:
                results.append(
                    SchemeResult(
                        name="Viterbi + Pseudo-Random Diagonal",
                        scheme_key="viterbi_diagonal",
                        rs_size=None,
                        confidence=conf,
                        decoded_bits=u_hat,
                        details=f"Re-encode consistency match: {100.0 * (1.0 - d):.1f}% (d_H={d*100:.1f}%)",
                    )
                )
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 7b. Viterbi + True Diagonal Interleaver (16x16 deterministic)
    # -------------------------------------------------------------------------
    if L >= 256:
        try:
            deint = true_diagonal_deinterleave(payload, rows=16, cols=16)
            u_hat = viterbi_decode(deint)
            c_re = conv_encode(u_hat, add_tail=True)
            re_int = true_diagonal_interleave(c_re, rows=16, cols=16)
            n_chk = min(L, len(re_int))
            d = float(np.mean(payload[:n_chk] != re_int[:n_chk]))
            conf = _viterbi_confidence(d, "viterbi_true_diagonal")
            if conf > 0.0:
                results.append(
                    SchemeResult(
                        name="Viterbi + True Diagonal",
                        scheme_key="viterbi_true_diagonal",
                        rs_size=None,
                        confidence=conf,
                        decoded_bits=u_hat,
                        details=f"Re-encode consistency match: {100.0 * (1.0 - d):.1f}% (d_H={d*100:.1f}%)",
                    )
                )
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 8. Concatenated (RS(64,48) + Viterbi)
    # -------------------------------------------------------------------------
    if L >= 1036:
        try:
            dec_bits = concatenated_decode(payload[:1036], n=64, k=48)
            re_cw = concatenated_encode(dec_bits, n=64, k=48)
            n_chk = min(L, len(re_cw))
            d = float(np.mean(payload[:n_chk] != re_cw[:n_chk]))
            if d < 0.12:
                conf = float(np.clip(1.0 - d * 2.0, 0.0, 1.0))
                results.append(
                    SchemeResult(
                        name="Concatenated (RS+Viterbi)",
                        scheme_key="concatenated",
                        rs_size=(64, 48),
                        confidence=conf,
                        decoded_bits=dec_bits,
                        details=f"RS(64,48)+Viterbi consistent (re-encode d_H={d*100:.1f}%)",
                    )
                )
        except Exception:
            pass

    # Sort by confidence descending.
    results.sort(key=lambda x: x.confidence, reverse=True)

    # Global confidence gate: if the best candidate doesn't clear the minimum
    # confidence floor, return an empty list rather than forcing a winner on
    # noise or uncorrelated data.  Callers should treat [] as "unidentified".
    if results and results[0].confidence < _MIN_CONFIDENCE:
        return []

    return results
