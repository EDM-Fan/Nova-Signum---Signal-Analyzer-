"""
samples/generate_concat_demo.py
===============================
Generates a BPSK demo .iq file encoded with Concatenated FEC
(Outer Reed-Solomon RS(64,48) + Inner Rate-1/2 Convolutional/Viterbi K=7),
for live GUI click-through verification of the Concatenated FEC scheme.

Packet structure:
  [32-bit SYNC word (0xEB902A3C)] + [Concatenated Codeword = 1036 bits]

Channel: BPSK, RRC pulse shaping, sps=8, Fs=1 MHz, SNR=15 dB, CFO=300 Hz, Phase=π rad.

Output files (relative to this script's directory):
  concat_bpsk_demo.iq          — float32 interleaved I/Q samples
  concat_bpsk_demo_bits.npy    — ground-truth 384-bit (48-byte) info payload
"""

import os
import sys
import numpy as np

# Project root on path (one level up)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modulation.concatenated import concatenated_encode
from modulation.correlator import DEFAULT_SYNC_32
from modulation.signal_gen import _rrc_filter, _apply_offsets, _add_awgn

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Concatenated scheme dimensions
N_RS, K_RS = 64, 48


def generate_concat_demo(
    sample_rate: int = 1_000_000,
    sps: int = 8,
    snr_db: float = 15.0,
    freq_offset_hz: float = 300.0,
    phase_offset_rad: float = 3.14159,
    seed: int = 42,
):
    """Generate and save a BPSK signal encoded with Concatenated RS(64,48)+Viterbi."""
    np.random.seed(seed)

    # 1. Generate 48 random info bytes (384 bits)
    info_bytes = np.random.randint(0, 256, K_RS, dtype=np.uint8)
    info_bits = np.unpackbits(info_bytes)

    # 2. Concatenated Encode: Outer RS(64,48) -> Inner Conv(Rate 1/2, K=7)
    codeword_bits = concatenated_encode(info_bytes, n=N_RS, k=K_RS)
    expected_cw_len = 2 * (N_RS * 8 + 6)  # 1036 bits
    assert len(codeword_bits) == expected_cw_len, f"Expected {expected_cw_len} bits, got {len(codeword_bits)}"

    # 3. Packetize: [32-bit SYNC] + [1036 Concatenated codeword bits] = 1068 packet bits
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
    iq_path = os.path.join(OUTPUT_DIR, "concat_bpsk_demo.iq")
    raw_iq = np.empty(2 * len(sig), dtype=np.float32)
    raw_iq[0::2] = sig.real.astype(np.float32)
    raw_iq[1::2] = sig.imag.astype(np.float32)
    raw_iq.tofile(iq_path)

    # 6. Save ground-truth info bits sidecar
    npy_path = os.path.join(OUTPUT_DIR, "concat_bpsk_demo_bits.npy")
    np.save(npy_path, info_bits)

    print("Concatenated (RS(64,48) + Viterbi) BPSK demo generator")
    print("=" * 60)
    print(f"  Outer Code          : Reed-Solomon RS({N_RS},{K_RS})  [t=8, GF(2^8)]")
    print(f"  Inner Code          : Convolutional Rate-1/2, K=7 (G1=171o, G2=133o)")
    print(f"  Info payload        : {len(info_bits)} bits ({len(info_bytes)} bytes)")
    print(f"  Codeword            : {len(codeword_bits)} bits")
    print(f"  Sync word           : 32 bits (0xEB902A3C)")
    print(f"  Total packet bits   : {len(packet_bits)} ({len(sync_word)} sync + {len(codeword_bits)} payload)")
    print(f"  Channel conditions  : SNR={snr_db} dB | CFO={freq_offset_hz} Hz | Phase={phase_offset_rad:.2f} rad")
    print(f"  IQ samples          : {len(sig):,} complex samples")
    print(f"  Output IQ file      : {os.path.basename(iq_path)}  ({os.path.getsize(iq_path):,} bytes)")
    print(f"  Ground truth sidecar: {os.path.basename(npy_path)}")
    print()
    print("Load concat_bpsk_demo.iq in the GUI, select Concatenated (RS+Viterbi)")
    print("from the FEC dropdown, then run: Classify -> Demodulate -> Find Sync -> Decode (FEC)")
    print("Expected label: 'Decoded (384 bits / 48 bytes [Concatenated RS(64,48)+Viterbi]):'")


if __name__ == "__main__":
    generate_concat_demo()
