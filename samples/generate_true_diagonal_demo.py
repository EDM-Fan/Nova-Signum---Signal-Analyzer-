"""
samples/generate_true_diagonal_demo.py
======================================
Generates a realistic BPSK IQ signal containing:
  - 32-bit sync preamble (DEFAULT_SYNC_32)
  - 250 information bits
  - Convolutionally encoded (K=7, Rate 1/2) -> 512 coded bits
  - True Deterministic Diagonal Interleaved (16x16 matrix cyclic-shift) -> 512 interleaved bits
  - 15 dB AWGN channel, CFO 300 Hz and 180 deg phase ambiguity

Used for end-to-end verification and live GUI testing of True Diagonal de-interleaving.
"""

import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modulation.correlator import DEFAULT_SYNC_32
from modulation.fec import conv_encode
from modulation.deinterleaver import true_diagonal_interleave
from modulation.signal_gen import _rrc_filter, _apply_offsets, _add_awgn

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_true_diagonal_demo(
    num_info_bits: int = 250,
    sample_rate: int = 1_000_000,
    sps: int = 8,
    snr_db: float = 15.0,
    freq_offset_hz: float = 300.0,
    phase_offset_rad: float = 3.14159,  # 180 deg phase ambiguity
    seed: int = 42,
) -> tuple[str, str]:
    np.random.seed(seed)

    # 1. Info bits (250 bits + 6 tail = 256 * 2 = 512 coded bits)
    info_bits = np.random.randint(0, 2, num_info_bits, dtype=np.uint8)

    # 2. Convolutional encode (250 -> 512 bits)
    coded = conv_encode(info_bits, add_tail=True)

    # 3. True Diagonal Interleave (16x16 -> 512 bits)
    interleaved = true_diagonal_interleave(coded, rows=16, cols=16)

    # 4. Frame: [Sync 32] + [Interleaved 512] = 544 bits
    sync_word = DEFAULT_SYNC_32.copy()
    frame_bits = np.concatenate([sync_word, interleaved])

    # 5. Modulate BPSK
    symbols = 2 * frame_bits.astype(float) - 1.0
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols

    h = _rrc_filter(beta=0.35, span=8, sps=sps)
    tx = np.convolve(upsampled, h, mode="same")

    # 6. Apply Channel
    rx = _apply_offsets(tx, sample_rate, freq_offset_hz, phase_offset_rad)
    rx = _add_awgn(rx, snr_db)

    # 7. Write IQ and ground-truth files
    iq_path = os.path.join(OUTPUT_DIR, "true_diagonal_bpsk_demo.iq")
    gt_path = os.path.join(OUTPUT_DIR, "true_diagonal_bpsk_demo_info_bits.npy")

    # Save as interleaved float32 IQ
    iq_interleaved = np.empty(len(rx) * 2, dtype=np.float32)
    iq_interleaved[0::2] = rx.real.astype(np.float32)
    iq_interleaved[1::2] = rx.imag.astype(np.float32)
    iq_interleaved.tofile(iq_path)

    np.save(gt_path, info_bits)

    print(f"Generated {iq_path} ({len(rx)} complex samples, {os.path.getsize(iq_path):,} bytes)")
    print(f"Saved ground-truth info bits ({len(info_bits)} bits) -> {gt_path}")
    return iq_path, gt_path


if __name__ == "__main__":
    generate_true_diagonal_demo()
