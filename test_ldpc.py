"""
test_ldpc.py
============
Comprehensive verification test suite for Low-Density Parity-Check (LDPC) FEC (modulation/ldpc.py).

Test sequence:
1. Unit Test 1: Zero-error decode (exact identity match).
2. Unit Test 2: Error-rate characterization under increasing bit-flip noise (0.5%, 1%, 2%, 5%, 10%).
3. Unit Test 3: Clean failure signaling on non-convergence (heavily corrupted input).
4. Unit Test 4: Full RF chain (Bits -> LDPC Encode -> BPSK Modulate -> AWGN/CFO/Phase -> Demod -> Sync -> LDPC Decode -> BER).
"""

import os
import sys
import numpy as np

# Project root on path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modulation.ldpc import (
    DEFAULT_H_128_64,
    DEFAULT_G_128_64,
    build_ldpc_code,
    ldpc_encode,
    ldpc_decode,
)
from modulation.correlator import find_sync, DEFAULT_SYNC_32
from modulation.demodulator import demodulate
from modulation.signal_gen import _rrc_filter, _apply_offsets, _add_awgn


def test_ldpc_zero_errors(n=128, k=64):
    """Test 1: Zero errors introduced — verify exact decode."""
    np.random.seed(42)
    msg = np.random.randint(0, 2, k, dtype=np.uint8)
    cw = ldpc_encode(msg)

    decoded, converged, iters = ldpc_decode(cw)
    match = np.array_equal(msg, decoded)

    print("=== Test 1: LDPC Zero Errors Introduced ===")
    print(f"  Code                : LDPC({n}, {k}) [Rate 1/2]")
    print(f"  Message length      : {len(msg)} bits")
    print(f"  Codeword length     : {len(cw)} bits")
    print(f"  Parity Checks Passed: {converged} (Iterations used: {iters})")
    print(f"  Exact Identity Match: {match}")
    assert match, "LDPC zero-error decode failed to match original message!"
    assert converged, "LDPC zero-error decode reported non-convergence!"
    print("  [PASS] Zero-error decode succeeded perfectly!\n")


def test_ldpc_error_characterization(n=128, k=64):
    """Test 2: Characterize LDPC performance curve across increasing raw channel bit-flip rates."""
    np.random.seed(123)
    ber_levels = [0.005, 0.01, 0.02, 0.05, 0.10]
    trials_per_level = 50

    print("=== Test 2: LDPC Bit-Flip Error Characterization Curve ===")
    print(f"  Code: LDPC({n}, {k}) Rate-1/2 | {trials_per_level} trials per noise level")
    print("  -------------------------------------------------------------------------")
    print("  Raw Channel BER | Convergence Rate | Avg Iterations | Decoded Residual BER")
    print("  -------------------------------------------------------------------------")

    for ber in ber_levels:
        successes = 0
        total_iters = []
        total_bit_errors = 0
        total_info_bits = trials_per_level * k

        for _ in range(trials_per_level):
            msg = np.random.randint(0, 2, k, dtype=np.uint8)
            cw = ldpc_encode(msg)

            # Flip bits with probability = ber
            noise = (np.random.rand(n) < ber).astype(np.uint8)
            rx = cw ^ noise

            decoded, conv, it = ldpc_decode(rx, max_iterations=50)
            if conv and np.array_equal(msg, decoded):
                successes += 1
            total_iters.append(it)
            total_bit_errors += np.sum(msg != decoded)

        conv_rate = (successes / trials_per_level) * 100.0
        avg_iter = np.mean(total_iters)
        res_ber = (total_bit_errors / total_info_bits) * 100.0

        print(f"       {ber*100:5.1f}%      |      {conv_rate:5.1f}%       |      {avg_iter:5.1f}     |       {res_ber:5.2f}%")

    print("  -------------------------------------------------------------------------")
    print("  [PASS] LDPC error characterization completed successfully!\n")


def test_ldpc_non_convergence_failure(n=128, k=64):
    """Test 3: Heavily corrupted input (50% random noise) — verify clean failure signaling."""
    np.random.seed(456)
    msg = np.random.randint(0, 2, k, dtype=np.uint8)
    cw = ldpc_encode(msg)

    # 50% bit flips (pure garbage)
    corrupted_cw = cw ^ np.random.randint(0, 2, n, dtype=np.uint8)

    print("=== Test 3: LDPC Non-Convergence Failure Signaling ===")
    
    # 1. Check boolean signaling
    decoded, converged, iters = ldpc_decode(corrupted_cw, max_iterations=20, raise_on_failure=False)
    print(f"  Boolean Signaling   : converged={converged}, iters={iters}")
    assert not converged, "Decoder falsely claimed convergence on 50% corrupted input!"

    # 2. Check exception signaling with raise_on_failure=True
    raised_cleanly = False
    exc_msg = ""
    try:
        ldpc_decode(corrupted_cw, max_iterations=20, raise_on_failure=True)
    except ValueError as exc:
        raised_cleanly = True
        exc_msg = str(exc)

    print(f"  Clean Exception     : {raised_cleanly}")
    print(f"  Exception Message   : {exc_msg}")
    assert raised_cleanly, "Decoder failed to raise ValueError when raise_on_failure=True!"
    print("  [PASS] Non-convergence cleanly signaled via both boolean and exception modes!\n")


def test_ldpc_rf_pipeline_bpsk(n=128, k=64, sps=8, fs=1_000_000, snr_db=15.0, cfo_hz=300.0, phase_rad=3.14159):
    """
    Test 4: Full RF Pipeline with LDPC FEC:
    Info Bits (64) -> LDPC Encode (128) -> Append Sync Word (32 bits) ->
    BPSK Modulate -> RRC Filter -> Channel Impairments (CFO 300Hz, Phase 180 deg, AWGN 15dB) ->
    Demodulate -> Find Sync (Correlator) -> Extract Payload -> LDPC Decode -> Check 0.00% BER.
    """
    np.random.seed(789)
    print("=== Test 4: End-to-End RF Pipeline with LDPC FEC (BPSK) ===")
    print(f"  Channel Profile: SNR = {snr_db:.1f} dB, CFO = {cfo_hz:+.0f} Hz, Phase = {phase_rad:.2f} rad, SPS = {sps}")

    # 1. Generate Info payload and LDPC codeword
    info_bits = np.random.randint(0, 2, k, dtype=np.uint8)
    cw_bits = ldpc_encode(info_bits)

    # 2. Frame Assembly: Sync Word (32 bits) + Codeword (128 bits)
    frame_bits = np.concatenate([DEFAULT_SYNC_32, cw_bits])

    # 3. BPSK Modulation (+1 -> 0, -1 -> 1 or 2*bits-1)
    symbols = 2 * frame_bits - 1
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols

    # 4. RRC Pulse Shaping & Channel Impairments
    h = _rrc_filter(beta=0.35, span=8, sps=sps)
    tx_signal = np.convolve(upsampled, h, mode="same")
    rx_impaired = _apply_offsets(tx_signal, fs, cfo_hz, phase_rad)
    rx_signal = _add_awgn(rx_impaired, snr_db)

    # 5. Demodulate
    raw_rx_bits = demodulate(rx_signal, fs, "BPSK", sps=sps)

    # 6. Sync Word Detection
    sync_res = find_sync(raw_rx_bits, sync_pattern=DEFAULT_SYNC_32, modulation_type="BPSK")
    assert sync_res["offset"] is not None, "Failed to detect sync word in demodulated bitstream!"
    print(f"  Sync Correlation    : Offset = {sync_res['offset']} | Match = {sync_res['score']*100:.1f}% ({sync_res['variant']})")

    # 7. Extract Payload & Decode LDPC
    payload_bits = sync_res["payload_bits"][:n]
    assert len(payload_bits) == n, f"Extracted payload length {len(payload_bits)} != codeword length {n}"

    decoded_info, converged, iters = ldpc_decode(payload_bits, max_iterations=50)
    assert converged, "LDPC decoder failed to converge on RF pipeline payload!"

    # 8. Compute Final Info BER
    bit_errors = np.sum(info_bits != decoded_info)
    info_ber = bit_errors / k * 100.0

    print(f"  Decoded Payload     : {len(decoded_info)} info bits (Iterations: {iters})")
    print(f"  Final Info BER      : {info_ber:.2f}% ({bit_errors} bit errors)")
    assert bit_errors == 0, f"Full RF pipeline failed to recover info bits (BER = {info_ber:.2f}%)!"
    print("  [PASS] Full RF pipeline with LDPC FEC recovered with 0.00% BER!\n")


if __name__ == "__main__":
    print("=================================================================")
    print("           RUNNING LDPC FEC VERIFICATION TEST SUITE              ")
    print("=================================================================\n")
    test_ldpc_zero_errors()
    test_ldpc_error_characterization()
    test_ldpc_non_convergence_failure()
    test_ldpc_rf_pipeline_bpsk()
    print("=================================================================")
    print("            ALL LDPC VERIFICATION TESTS PASSED!                  ")
    print("=================================================================")
