"""
test_concatenated.py
====================
Unit tests and demonstration for Concatenated Forward Error Correction
(Outer Reed-Solomon RS(64,48) + Inner Convolutional / Viterbi Rate-1/2, K=7).

NASA / CCSDS 131.0-B Architecture:
1. Zero-error encode / decode roundtrip.
2. Direct comparison demonstrating concatenation outperforming Viterbi-alone at
   p = 5% channel error rate (where Viterbi leaves residual errors, but RS mops them up).
3. Propagation of clean uncorrectable-error signaling (ValueError) on severe channel corruption.
4. Full RF pipeline test: BPSK + RRC filtering + AWGN (15 dB) + CFO (300 Hz) + Phase Inversion (pi).
"""

import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modulation.concatenated import concatenated_encode, concatenated_decode
from modulation.fec import viterbi_decode
from modulation.rs_fec import rs_encode, rs_decode
from modulation.correlator import find_sync, DEFAULT_SYNC_32
from modulation.demodulator import demodulate
from modulation.signal_gen import _rrc_filter, _apply_offsets, _add_awgn


def test_zero_error_roundtrip():
    """Test 1: Zero-error encode / decode roundtrip with byte and bit inputs."""
    print("=== Test 1: Zero-Error Concatenated FEC Roundtrip ===")
    np.random.seed(42)

    # 1A. Byte input (48 bytes = 384 bits)
    info_bytes = np.random.randint(0, 256, 48, dtype=np.uint8)
    coded_bits = concatenated_encode(info_bytes, n=64, k=48)
    expected_len = 2 * (64 * 8 + 6)  # 1036 bits
    assert len(coded_bits) == expected_len, f"Encoded length mismatch: expected {expected_len}, got {len(coded_bits)}"

    decoded_bits = concatenated_decode(coded_bits, n=64, k=48)
    expected_bits = np.unpackbits(info_bytes)
    assert np.array_equal(decoded_bits, expected_bits), "Decoded bits mismatch from byte input"
    print(f"  [PASS] 48 information bytes -> {len(coded_bits)} channel bits -> 384 decoded bits (0 errors).")

    # 1B. Bit input (384 bits)
    info_bits = np.unpackbits(info_bytes)
    coded_bits_from_bits = concatenated_encode(info_bits, n=64, k=48)
    assert np.array_equal(coded_bits, coded_bits_from_bits)

    decoded_bits_2 = concatenated_decode(coded_bits_from_bits, n=64, k=48)
    assert np.array_equal(decoded_bits_2, info_bits), "Decoded bits mismatch from bit input"
    print(f"  [PASS] 384 information bits -> {len(coded_bits_from_bits)} channel bits -> 384 decoded bits (0 errors).\n")


def test_concatenation_vs_viterbi_alone():
    """
    Test 2: Key demonstration — channel error rate (p = 5.0%) where Viterbi-alone
    fails to achieve zero error (leaving ~1% residual BER), but Concatenation
    (Viterbi inner + RS outer) achieves 0.00% BER across 50 independent trials.
    """
    print("=== Test 2: Concatenation Outperforming Viterbi-Alone (p = 5.0% BSC) ===")
    n_trials = 50
    p_err = 0.05  # 5% channel bit error rate
    np.random.seed(12345)

    viterbi_err_bits_total = 0
    viterbi_total_bits = 0
    concat_err_bits_total = 0
    concat_total_bits = 0
    concat_success_count = 0
    viterbi_success_count = 0

    for trial in range(n_trials):
        info_bytes = np.random.randint(0, 256, 48, dtype=np.uint8)
        info_bits = np.unpackbits(info_bytes)

        # Concatenated encode: Outer RS(64,48) -> Inner Conv(Rate 1/2, K=7)
        coded_bits = concatenated_encode(info_bytes, n=64, k=48)

        # Binary Symmetric Channel with p = 5% bit error rate
        noise_mask = (np.random.rand(len(coded_bits)) < p_err).astype(np.uint8)
        rx_channel_bits = coded_bits ^ noise_mask

        # 1. Inner Viterbi alone:
        viterbi_out = viterbi_decode(rx_channel_bits)
        # First 384 bits correspond to info payload
        viterbi_info = viterbi_out[:384]
        v_errs = int(np.sum(viterbi_info != info_bits))
        viterbi_err_bits_total += v_errs
        viterbi_total_bits += 384
        if v_errs == 0:
            viterbi_success_count += 1

        # 2. Concatenated (Inner Viterbi + Outer RS):
        try:
            concat_decoded = concatenated_decode(rx_channel_bits, n=64, k=48)
            c_errs = int(np.sum(concat_decoded != info_bits))
            concat_err_bits_total += c_errs
            if c_errs == 0:
                concat_success_count += 1
        except ValueError:
            # RS declared uncorrectable error
            concat_err_bits_total += 384

        concat_total_bits += 384

    viterbi_ber = viterbi_err_bits_total / viterbi_total_bits
    concat_ber = concat_err_bits_total / concat_total_bits

    print(f"  Channel Bit Error Rate (BSC): {p_err * 100:.1f}%")
    print(f"  Trials: {n_trials} independent packets (48 bytes / 384 bits each)")
    print(f"  ----------------------------------------------------------------")
    print(f"  Viterbi-Alone:  BER = {viterbi_ber*100:5.2f}% | Packet Success = {viterbi_success_count}/{n_trials}")
    print(f"  Concatenated:   BER = {concat_ber*100:5.2f}% | Packet Success = {concat_success_count}/{n_trials}")

    # Assertions proving concatenation outperforms Viterbi alone
    assert viterbi_ber > 0.005, f"Viterbi alone unexpectedly had no errors (BER={viterbi_ber:.4f})"
    assert concat_ber == 0.0, f"Concatenated decoding failed (BER={concat_ber:.4f})"
    assert concat_success_count == n_trials, f"Concatenated success rate was {concat_success_count}/{n_trials}"
    print("  [PASS] Concatenation successfully eliminated 100% of residual Viterbi errors!\n")


def test_uncorrectable_error_signaling():
    """Test 3: Clean propagation of RS uncorrectable-error signaling (ValueError)."""
    print("=== Test 3: Uncorrectable Error Signaling ===")
    np.random.seed(999)

    # Random pure noise received bits
    noise_bits = np.random.randint(0, 2, 1036, dtype=np.uint8)
    error_caught = False
    try:
        concatenated_decode(noise_bits, n=64, k=48)
    except ValueError as e:
        error_caught = True
        print(f"  [PASS] Correctly raised ValueError on uncorrectable channel noise: {e}\n")

    assert error_caught, "Expected ValueError was not raised for pure noise received bits"


def test_noisy_rf_pipeline():
    """
    Test 4: Full RF Pipeline End-to-End Test.
    Transmitter: Sync (32 bits) + Concatenated RS(64,48)+Conv Payload (1036 bits) -> BPSK -> RRC.
    Channel: AWGN (SNR 15 dB) + CFO (300 Hz) + Phase Inversion (pi rad).
    Receiver: demodulate() -> find_sync() -> concatenated_decode() -> verify 0 bit errors.
    """
    print("=== Test 4: Full RF Pipeline Test (BPSK + AWGN 15dB + CFO 300Hz + Phase Inversion) ===")
    np.random.seed(42)
    fs = 1_000_000
    sps = 8
    snr_db = 15.0
    cfo_hz = 300.0
    phase_rad = 3.1415926535

    # 1. Generate information payload (48 bytes = 384 bits)
    info_bytes = np.random.randint(0, 256, 48, dtype=np.uint8)
    info_bits = np.unpackbits(info_bytes)

    # 2. Encode concatenated payload (1036 channel bits)
    payload_bits = concatenated_encode(info_bytes, n=64, k=48)

    # 3. Assemble frame: 32-bit sync + 1036-bit payload (1068 bits total)
    frame_bits = np.concatenate([DEFAULT_SYNC_32, payload_bits])

    # 4. Modulate to BPSK baseband with RRC pulse shaping
    symbols = 2 * frame_bits.astype(float) - 1.0
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    h = _rrc_filter(beta=0.35, span=8, sps=sps)
    tx_clean = np.convolve(up, h, mode="same")

    # 5. Channel impairments
    tx_cfo = _apply_offsets(tx_clean, fs, cfo_hz, phase_rad)
    rx_noisy = _add_awgn(tx_cfo, snr_db=snr_db)

    # 6. Receiver pipeline
    demod_bits = demodulate(rx_noisy, fs, "BPSK", sps=sps)
    print(f"  Demodulated: {len(demod_bits)} raw channel bits.")

    # 7. Frame synchronization
    sync_res = find_sync(demod_bits, sync_pattern=DEFAULT_SYNC_32, modulation_type="BPSK")
    assert sync_res["score"] >= 0.8, f"Sync word not found in demodulated stream (score: {sync_res['score']})"
    print(f"  Sync detected: offset={sync_res['offset']}, variant={sync_res['variant']}, score={sync_res['score']}")

    rx_payload = sync_res["payload_bits"]
    assert len(rx_payload) >= 1036, f"Payload too short: {len(rx_payload)} < 1036"

    # 8. Concatenated FEC decoding
    decoded_bits = concatenated_decode(rx_payload[:1036], n=64, k=48)

    # 9. Verification
    bit_errors = int(np.sum(decoded_bits != info_bits))
    print(f"  Decoded payload length: {len(decoded_bits)} bits.")
    print(f"  Bit errors against ground truth: {bit_errors} / {len(info_bits)} (BER = {bit_errors/len(info_bits):.4f})")
    assert bit_errors == 0, f"Expected 0 bit errors, got {bit_errors}"
    print("  [PASS] Full RF pipeline successfully recovered 100% of payload bits!\n")


if __name__ == "__main__":
    test_zero_error_roundtrip()
    test_concatenation_vs_viterbi_alone()
    test_uncorrectable_error_signaling()
    test_noisy_rf_pipeline()
    print("================================================================")
    print("ALL CONCATENATED FEC UNIT TESTS PASSED SUCCESSFULLY!")
    print("================================================================")
