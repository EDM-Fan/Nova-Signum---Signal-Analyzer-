"""
samples/generate_16qam_demo.py
==============================
Generates a 16QAM demo signal with sync word and ground truth payload.
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modulation.correlator import DEFAULT_SYNC_32
from modulation.signal_gen import generate_16qam_from_bits

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_16qam_demo(n_payload_bits=7968, sample_rate=1000000, sps=8, snr_db=18.0):
    np.random.seed(42)
    sync_word = DEFAULT_SYNC_32
    payload = np.random.randint(0, 2, n_payload_bits)
    packet = np.concatenate([sync_word, payload])
    sig = generate_16qam_from_bits(packet, sps=sps, sample_rate=sample_rate, snr_db=snr_db)
    iq_path = os.path.join(OUTPUT_DIR, '16qam_demo.iq')
    raw_iq = np.empty(2 * len(sig), dtype=np.float32)
    raw_iq[0::2] = sig.real.astype(np.float32)
    raw_iq[1::2] = sig.imag.astype(np.float32)
    raw_iq.tofile(iq_path)
    npy_path = os.path.join(OUTPUT_DIR, '16qam_demo_bits.npy')
    np.save(npy_path, payload)
    print(f"Saved 16QAM demo: {iq_path} ({os.path.getsize(iq_path):,} bytes) and {npy_path}")
    return iq_path, npy_path


if __name__ == '__main__':
    generate_16qam_demo()
