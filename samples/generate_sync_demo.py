"""
samples/generate_sync_demo.py
==============================
Generates a minimal BPSK signal for testing the "Find Sync" button in the GUI.

Signal structure:
  1. 32-bit sync word  (DEFAULT_SYNC_32 = 0xEB902A3C, unencoded/uninterleaved)
  2. 384 random payload bits  (chosen to be clearly larger than the sync word)
  ─────────────────────────────
  3. Total packet: 416 bits

Modulation: BPSK, RRC pulse shaping (β=0.35, span=8, sps=8)
Channel:    18 dB SNR AWGN, zero CFO/phase offset (clean, deterministic)
Fs:         1 MHz (float32 interleaved IQ)

Output:
  samples/bpsk_sync_demo.iq        – float32 interleaved I/Q samples
  samples/bpsk_sync_demo_bits.npy  – ground-truth 32-bit sync word (for verification)

Expected GUI result after loading bpsk_sync_demo.iq with Fs=1MHz:
  Classify  → BPSK
  Demodulate → 416+ bits recovered
  Find Sync  → sync @ bit 0, score ≥ 95%, variant = "DEFAULT_SYNC_32"
"""

import os
import sys
import numpy as np

# ── project root ──────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modulation.correlator import DEFAULT_SYNC_32
from modulation.signal_gen import _rrc_filter, _apply_offsets, _add_awgn

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_sync_demo(
    n_payload_bits: int = 384,
    sample_rate: int = 1_000_000,
    sps: int = 8,
    snr_db: float = 18.0,
    freq_offset_hz: float = 0.0,
    phase_offset_rad: float = 0.0,
    seed: int = 7,
) -> tuple[str, str]:
    """Generate and save the sync demo IQ file.

    Returns
    -------
    iq_path, npy_path : str
        Absolute paths to the saved files.
    """
    np.random.seed(seed)

    # 1. Sync word (always first 32 bits)
    sync_word = DEFAULT_SYNC_32.copy()   # shape (32,), dtype int/bool

    # 2. Random payload
    payload = np.random.randint(0, 2, n_payload_bits)

    # 3. Concatenate → packet bits
    packet_bits = np.concatenate([sync_word, payload])

    # 4. BPSK modulation: 0→-1, 1→+1
    symbols = 2.0 * packet_bits - 1.0  # float64 ±1

    # 5. Upsample + RRC pulse shaping
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols

    h = _rrc_filter(beta=0.35, span=8, sps=sps)
    sig = np.convolve(upsampled, h, mode="same")

    # 6. Channel impairments
    sig = _apply_offsets(sig, sample_rate, freq_offset_hz, phase_offset_rad)
    sig = _add_awgn(sig, snr_db)

    # 7. Save as float32 interleaved IQ
    iq_path = os.path.join(OUTPUT_DIR, "bpsk_sync_demo.iq")
    raw_iq = np.empty(2 * len(sig), dtype=np.float32)
    raw_iq[0::2] = sig.real.astype(np.float32)
    raw_iq[1::2] = sig.imag.astype(np.float32)
    raw_iq.tofile(iq_path)

    # 8. Save ground-truth sync word (for offline verification)
    npy_path = os.path.join(OUTPUT_DIR, "bpsk_sync_demo_bits.npy")
    np.save(npy_path, sync_word)

    print("Generated sync demo:")
    print(f"  Sync word bits   : {len(sync_word)}  (DEFAULT_SYNC_32 = 0xEB902A3C)")
    print(f"  Payload bits     : {n_payload_bits}")
    print(f"  Total packet     : {len(packet_bits)} bits")
    print(f"  IQ samples       : {len(sig):,}")
    print(f"  SNR              : {snr_db} dB  |  CFO: {freq_offset_hz} Hz")
    print(f"  Saved IQ file    : {iq_path}  ({os.path.getsize(iq_path):,} bytes)")
    print(f"  Saved ground truth: {npy_path}")
    print()
    print("Expected GUI workflow:")
    print("  1. Open bpsk_sync_demo.iq  (Fs = 1 MHz)")
    print("  2. Classify Modulation     -> BPSK")
    print("  3. Demodulate              -> ~416+ bits")
    print('  4. Find Sync               -> sync @ bit 0, >= 95% match, variant "DEFAULT_SYNC_32"')

    return iq_path, npy_path


if __name__ == "__main__":
    generate_sync_demo()
