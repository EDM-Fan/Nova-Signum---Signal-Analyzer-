"""
samples/generate_altb_sync_demo.py
====================================
Generates a BPSK test packet encoded with SYNC_ALT_B_32 (0xFAF334BE)
as the sync word — distinct from DEFAULT_SYNC_32 (0xEB902A3C) and
SYNC_CCSDS_32 (0x1ACFFC1D) — to verify that the cmb_sync selector in
the GUI is functionally doing different things for each selection.

Signal structure (same pipeline as bpsk_sync_alt_demo.iq / Alt A):
  1. 512 random information bits
  2. Convolutional encoding (K=7, Rate 1/2) → 1036 coded bits
  3. Pad to 1280 (5 × 256-bit blocks)
  4. Block interleaving (8 × 32) → 1280 interleaved bits
  5. Prepend 32-bit SYNC_ALT_B_32 (0xFAF334BE) → 1312 packet bits
  6. BPSK modulation, RRC pulse shaping (β=0.35, span=8, sps=8)
  7. Channel: SNR=18 dB, CFO=0 Hz, no phase offset (clean + deterministic)
  8. Saved as float32 interleaved IQ  → bpsk_altb_sync_demo.iq
  9. Ground-truth info bits saved     → bpsk_altb_sync_demo_bits.npy

GUI verification:
  - With "Alt B (0xFAF334BE)" selected  → sync found at offset 0, ≥ 95% match
  - With "Default (0xEB902A3C)" selected → match < 20% (no genuine sync)
"""

import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modulation.fec import conv_encode
from modulation.deinterleaver import block_interleave
from modulation.correlator import SYNC_ALT_B_32
from modulation.signal_gen import _rrc_filter, _apply_offsets, _add_awgn

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_altb_sync_demo(
    num_info_bits: int = 512,
    sample_rate: int = 1_000_000,
    sps: int = 8,
    snr_db: float = 18.0,
    freq_offset_hz: float = 0.0,
    phase_offset_rad: float = 0.0,
    seed: int = 77,
) -> tuple[str, str]:
    """Generate and save the Alt-B sync demo IQ file.

    Returns
    -------
    iq_path, npy_path : str
        Absolute paths to the saved files.
    """
    np.random.seed(seed)

    # 1. 512 information bits
    info_bits = np.random.randint(0, 2, num_info_bits)

    # 2. Convolutional encoding (512 info + 6 tail → 1036 coded bits)
    coded_bits = conv_encode(info_bits)

    # 3. Pad to multiple of 256 (5 blocks = 1280 bits)
    block_size = 8 * 32   # 256
    n_pad = (block_size - (len(coded_bits) % block_size)) % block_size
    dummy_pad = np.random.randint(0, 2, n_pad)
    padded_coded = np.concatenate([coded_bits, dummy_pad])

    # 4. Block interleaving (1280 bits, 5 × 8×32 blocks)
    interleaved_bits = block_interleave(padded_coded, rows=8, cols=32)

    # 5. Prepend SYNC_ALT_B_32 (0xFAF334BE)
    sync_word = SYNC_ALT_B_32.copy()
    packet_bits = np.concatenate([sync_word, interleaved_bits])

    # 6. BPSK modulation: 0→-1, 1→+1
    symbols = 2 * packet_bits - 1.0
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols

    h = _rrc_filter(beta=0.35, span=8, sps=sps)
    sig = np.convolve(upsampled, h, mode="same")
    sig = _apply_offsets(sig, sample_rate, freq_offset_hz, phase_offset_rad)
    sig = _add_awgn(sig, snr_db)

    # 7. Save float32 interleaved IQ
    iq_path = os.path.join(OUTPUT_DIR, "bpsk_altb_sync_demo.iq")
    raw_iq = np.empty(2 * len(sig), dtype=np.float32)
    raw_iq[0::2] = sig.real.astype(np.float32)
    raw_iq[1::2] = sig.imag.astype(np.float32)
    raw_iq.tofile(iq_path)

    # 8. Save ground-truth info bits
    npy_path = os.path.join(OUTPUT_DIR, "bpsk_altb_sync_demo_bits.npy")
    np.save(npy_path, info_bits)

    print("Generated Alt-B sync demo packet:")
    print(f"  Sync Word         : Alt B (0xFAF334BE) [32 bits]")
    print(f"  Info Bits         : {num_info_bits}")
    print(f"  Coded Bits        : {len(coded_bits)}")
    print(f"  Interleaved Bits  : {len(interleaved_bits)}")
    print(f"  Total Packet Bits : {len(packet_bits)}  ({len(sync_word)} sync + {len(interleaved_bits)} payload)")
    print(f"  Complex IQ Samples: {len(sig):,}  ({os.path.getsize(iq_path):,} bytes)")
    print(f"  Saved IQ File     : {iq_path}")
    print(f"  Saved Ground Truth: {npy_path}")

    return iq_path, npy_path


if __name__ == "__main__":
    generate_altb_sync_demo()
