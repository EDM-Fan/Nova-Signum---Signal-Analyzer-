"""
samples/generate_ldpc_demo.py
=============================
Generates a BPSK demo .iq file encoded with Low-Density Parity-Check (LDPC) FEC,
for live GUI click-through verification of the LDPC FEC scheme.

Packet structure:
  [32-bit SYNC word (0xEB902A3C)] + [LDPC(128,64) codeword = 128 bits]

Channel: BPSK, RRC pulse shaping, sps=8, Fs=1 MHz, SNR=15 dB, CFO=300 Hz, Phase=π rad.

Output files (relative to this script's directory):
  ldpc_bpsk_demo.iq          — float32 interleaved I/Q samples
  ldpc_bpsk_demo_bits.npy    — ground-truth 64-bit info payload
"""

import os
import sys
import numpy as np

# Project root on path (one level up)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modulation.ldpc import ldpc_encode
from modulation.correlator import DEFAULT_SYNC_32
from modulation.signal_gen import _rrc_filter, _apply_offsets, _add_awgn

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# LDPC(128, 64) Rate-1/2
N, K = 128, 64


def generate_ldpc_demo(
    sample_rate: int = 1_000_000,
    sps: int = 8,
    snr_db: float = 15.0,
    freq_offset_hz: float = 300.0,
    phase_offset_rad: float = 3.14159,
    seed: int = 42,
):
    """Generate and save a BPSK signal encoded with LDPC(128,64)."""
    np.random.seed(seed)

    # 1. Generate 64 random info bits
    info_bits = np.random.randint(0, 2, K, dtype=np.uint8)

    # 2. LDPC(128,64) Encode: 64 bits -> 128 coded bits
    codeword_bits = ldpc_encode(info_bits)
    assert len(codeword_bits) == N, f"Expected {N} bits, got {len(codeword_bits)}"

    # 3. Packetize: [32-bit SYNC] + [128 LDPC codeword bits] = 160 packet bits
    sync_word = DEFAULT_SYNC_32
    packet_bits = np.concatenate([sync_word, codeword_bits])

    # 4. Modulate BPSK with RRC pulse shaping
    symbols = 2 * packet_bits - 1  # map 0->-1, 1->+1
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols
    h = _rrc_filter(beta=0.35, span=8, sps=sps)
    sig = np.convolve(upsampled, h, mode="same")
    sig = _apply_offsets(sig, sample_rate, freq_offset_hz, phase_offset_rad)
    sig = _add_awgn(sig, snr_db)

    # 5. Save float32 interleaved I/Q file
    iq_path = os.path.join(OUTPUT_DIR, "ldpc_bpsk_demo.iq")
    raw_iq = np.empty(2 * len(sig), dtype=np.float32)
    raw_iq[0::2] = sig.real.astype(np.float32)
    raw_iq[1::2] = sig.imag.astype(np.float32)
    raw_iq.tofile(iq_path)

    # 6. Save ground-truth info bits sidecar
    npy_path = os.path.join(OUTPUT_DIR, "ldpc_bpsk_demo_bits.npy")
    np.save(npy_path, info_bits)

    print("LDPC(128,64) BPSK demo generator")
    print("=" * 50)
    print(f"  LDPC code size      : LDPC({N},{K})  [Rate-1/2, {N-K} parity bits]")
    print(f"  Info payload        : {len(info_bits)} bits ({len(info_bits)//8} bytes)")
    print(f"  Codeword            : {len(codeword_bits)} bits ({len(codeword_bits)//8} bytes)")
    print(f"  Sync word           : 32 bits (0xEB902A3C)")
    print(f"  Total packet bits   : {len(packet_bits)} ({len(sync_word)} sync + {len(codeword_bits)} LDPC)")
    print(f"  Channel conditions  : SNR={snr_db} dB | CFO={freq_offset_hz} Hz | Phase={phase_offset_rad:.2f} rad")
    print(f"  IQ samples          : {len(sig):,} complex samples")
    print(f"  Output IQ file      : {os.path.basename(iq_path)}  ({os.path.getsize(iq_path):,} bytes)")
    print(f"  Ground truth sidecar: {os.path.basename(npy_path)}")
    print()
    print("Load ldpc_bpsk_demo.iq in the GUI, select LDPC (128,64) from the FEC")
    print("dropdown, then run: Classify -> Demodulate -> Find Sync -> Decode (FEC)")
    print("Expected label: 'Decoded (64 bits [LDPC(128,64), conv=True, it=0]):'")


if __name__ == "__main__":
    generate_ldpc_demo()
