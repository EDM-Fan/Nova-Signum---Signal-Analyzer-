"""
test_rs_fec.py
==============
Comprehensive verification test suite for Reed-Solomon FEC (modulation/rs_fec.py).

Test sequence in order of increasing complexity:
1. Unit Test 1: Zero errors (exact identity match).
2. Unit Test 2: Exactly maximum correctable byte errors (t = 8 for RS(64, 48)).
3. Unit Test 3: Over-capacity byte errors (t + 1 = 9) verifying clean failure (no silent corruption).
4. Unit Test 4: Full RF chain (Bits -> Bytes -> RS Encode -> BPSK Modulate -> AWGN/CFO/Phase -> Demod -> Sync -> RS Decode -> BER).
"""

import os
import sys
import numpy as np

# Project root on path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modulation.rs_fec import rs_encode, rs_decode
from modulation.correlator import find_sync, DEFAULT_SYNC_32
from modulation.demodulator import demodulate
from modulation.signal_gen import _rrc_filter, _apply_offsets, _add_awgn


def test_rs_zero_errors(n=64, k=48):
    """Test 1: Zero errors introduced — verify exact decode."""
    np.random.seed(42)
    msg = np.random.randint(0, 256, k, dtype=np.uint8)
    cw = rs_encode(msg, n=n, k=k)

    decoded = rs_decode(cw, n=n, k=k)
    match = np.array_equal(msg, decoded)

    print("=== Test 1: Zero Errors Introduced ===")
    print(f"  Code                : RS({n}, {k}) [Parity: {n-k} bytes, t = {(n-k)//2} errors]")
    print(f"  Message length      : {len(msg)} bytes ({len(msg)*8} bits)")
    print(f"  Codeword length     : {len(cw)} bytes ({len(cw)*8} bits)")
    print(f"  Exact Identity Match: {match}")
    assert match, "RS zero-error decode failed to match original message!"
    print("  [PASS] Zero-error decode succeeded perfectly!\n")


def test_rs_max_correctable_errors(n=64, k=48):
    """Test 2: Introduce exactly maximum correctable errors (t = 8) — verify exact recovery."""
    np.random.seed(123)
    t = (n - k) // 2
    msg = np.random.randint(0, 256, k, dtype=np.uint8)
    cw = rs_encode(msg, n=n, k=k)

    # Pick exactly t distinct byte positions to corrupt
    corrupt_indices = np.random.choice(n, size=t, replace=False)
    corrupted_cw = cw.copy()
    for idx in corrupt_indices:
        corrupted_cw[idx] ^= np.random.randint(1, 256, dtype=np.uint8)

    decoded = rs_decode(corrupted_cw, n=n, k=k)
    match = np.array_equal(msg, decoded)

    print("=== Test 2: Maximum Correctable Errors (t = 8 Byte Errors) ===")
    print(f"  Corrupted positions : {sorted(corrupt_indices.tolist())} (total {t} byte errors)")
    print(f"  Raw Byte Error Rate : {t / n * 100:.2f}% ({t * 8} channel bits corrupted)")
    print(f"  Exact Recovery Match: {match}")
    assert match, f"RS decode failed to correct {t} byte errors!"
    print("  [PASS] Maximum correctable errors fully recovered!\n")


def test_rs_over_capacity_errors(n=64, k=48):
    """Test 3: Introduce t + 1 = 9 byte errors — verify clean failure detection."""
    np.random.seed(456)
    t = (n - k) // 2
    num_errs = t + 1
    msg = np.random.randint(0, 256, k, dtype=np.uint8)
    cw = rs_encode(msg, n=n, k=k)

    corrupt_indices = np.random.choice(n, size=num_errs, replace=False)
    corrupted_cw = cw.copy()
    for idx in corrupt_indices:
        corrupted_cw[idx] ^= np.random.randint(1, 256, dtype=np.uint8)

    print("=== Test 3: Over-Capacity Errors (t + 1 = 9 Byte Errors) ===")
    print(f"  Corrupted positions : {sorted(corrupt_indices.tolist())} (total {num_errs} byte errors)")
    
    raised_cleanly = False
    error_msg = ""
    try:
        rs_decode(corrupted_cw, n=n, k=k)
    except ValueError as exc:
        raised_cleanly = True
        error_msg = str(exc)

    print(f"  Clean Failure Caught: {raised_cleanly}")
    print(f"  Decoder Exception   : {error_msg}")
    assert raised_cleanly, "Decoder silently returned corrupted data instead of raising an uncorrectable error!"
    print("  [PASS] Over-capacity error cleanly rejected without silent corruption!\n")


def test_full_pipeline_rs_bpsk(n=64, k=48, sps=8, fs=1_000_000, snr_db=15.0, cfo_hz=300.0, phase_rad=3.14159):
    """Test 4: Full RF pipeline using Reed-Solomon FEC with BPSK modulation, CFO, and 180-deg phase offset."""
    np.random.seed(789)
    # 1. Generate k random bytes (48 bytes = 384 info bits)
    info_bytes = np.random.randint(0, 256, k, dtype=np.uint8)
    info_bits = np.unpackbits(info_bytes)

    # 2. RS Encode: 48 bytes -> 64 bytes (512 coded bits)
    codeword_bytes = rs_encode(info_bytes, n=n, k=k)
    codeword_bits = np.unpackbits(codeword_bytes)
    assert len(codeword_bits) == n * 8

    # 3. Prepend 32-bit sync pattern: [32-bit SYNC] + [512 RS codeword bits]
    sync = DEFAULT_SYNC_32
    packet_bits = np.concatenate([sync, codeword_bits])

    # 4. Modulate BPSK with RRC pulse shaping
    symbols = 2 * packet_bits - 1
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols
    h = _rrc_filter(0.35, 8, sps)
    sig = np.convolve(upsampled, h, mode="same")
    sig = _apply_offsets(sig, fs, cfo_hz, phase_rad)
    sig = _add_awgn(sig, snr_db)

    # 5. Demodulate
    rec_bits = demodulate(sig, fs, "BPSK", sps=sps)

    # 6. Sync correlation and 180-degree phase ambiguity resolution
    sync_res = find_sync(rec_bits, sync_pattern=sync, modulation_type="BPSK")
    payload_rec = sync_res["payload_bits"][: len(codeword_bits)]

    # 7. Pack received bits to bytes and RS Decode
    rec_codeword_bytes = np.packbits(payload_rec)
    decoded_bytes = rs_decode(rec_codeword_bytes, n=n, k=k)
    decoded_bits = np.unpackbits(decoded_bytes)

    # 8. BER verification on original info bits
    bit_errors = np.sum(info_bits != decoded_bits)
    ber = float(bit_errors / len(info_bits))

    print("=== Test 4: Full RF Pipeline (RS(64, 48) + BPSK + AWGN + CFO + Phase Inversion) ===")
    print(f"  Info Payload        : {len(info_bytes)} bytes ({len(info_bits)} bits)")
    print(f"  Channel Packet      : {len(sync)} sync bits + {len(codeword_bits)} RS bits = {len(packet_bits)} bits")
    print(f"  Channel Conditions  : SNR = {snr_db} dB | CFO = {cfo_hz} Hz | Phase = {phase_rad:.2f} rad")
    print(f"  Sync Correlation    : Offset = {sync_res['offset']} | Match = {sync_res['score']*100:.1f}% ({sync_res['variant']})")
    print(f"  Decoded Info Bits   : {len(decoded_bits)} bits")
    print(f"  Final Bit Error Rate: {ber*100:.2f}% ({bit_errors} bit errors)")
    assert sync_res["offset"] == 0, f"Sync offset non-zero: {sync_res['offset']}"
    assert sync_res["score"] >= 0.95, f"Sync score too low: {sync_res['score']}"
    assert ber == 0.0, f"RS Pipeline BER non-zero: {ber}"
    print("  [PASS] Full RF Pipeline with Reed-Solomon recovered with 0.00% BER!\n")


if __name__ == "__main__":
    print("=================================================================")
    print("       RUNNING REED-SOLOMON FEC VERIFICATION TEST SUITE         ")
    print("=================================================================")

    RS_SIZES = [(32, 24), (64, 48), (128, 112)]

    for _n, _k in RS_SIZES:
        print(f"\n--- RS({_n},{_k}) unit tests ---")
        test_rs_zero_errors(n=_n, k=_k)
        test_rs_max_correctable_errors(n=_n, k=_k)
        test_rs_over_capacity_errors(n=_n, k=_k)

    # Full RF pipeline test kept at the canonical RS(64,48) size
    print("\n--- Full RF Pipeline (RS(64,48) + BPSK + AWGN) ---")
    test_full_pipeline_rs_bpsk(n=64, k=48)

    print("="*65)
    print("  ALL REED-SOLOMON VERIFICATION TESTS PASSED (0.00% BER)")
    print("="*65)
