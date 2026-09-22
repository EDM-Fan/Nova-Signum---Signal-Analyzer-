"""
test_fec_identifier.py
======================
Comprehensive verification test suite for Blind FEC / Interleaving Scheme Identification
(modulation/fec_identifier.py).

Test sequence:
1. Unit Tests: Clean (noiseless) identification for all 7 supported schemes:
   - Viterbi + Block (8x32)
   - Viterbi + Conv (8x4)
   - Viterbi + Diagonal
   - Reed-Solomon RS(32,24)
   - Reed-Solomon RS(64,48)
   - Reed-Solomon RS(128,112)
   - LDPC (128,64)
2. Noisy Channel RF Pipeline Tests: Blind identification under realistic channel conditions
   (BPSK + AWGN 15 dB + CFO 300 Hz + Phase offset):
   - Noisy Reed-Solomon RS(64,48)
   - Noisy LDPC (128,64)
   - Noisy Viterbi + Block (8x32)
3. Edge Case: Graceful handling of short / invalid payloads.
"""

import os
import sys
import numpy as np

# Project root on path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modulation.fec_identifier import identify_fec_scheme, SchemeResult
from modulation.fec import conv_encode, viterbi_decode
from modulation.deinterleaver import (
    block_interleave,
    conv_interleave,
    diagonal_interleave,
    true_diagonal_interleave,
)
from modulation.rs_fec import rs_encode
from modulation.ldpc import ldpc_encode
from modulation.concatenated import concatenated_encode
from modulation.correlator import find_sync, DEFAULT_SYNC_32
from modulation.demodulator import demodulate
from modulation.signal_gen import _rrc_filter, _apply_offsets, _add_awgn


def test_noiseless_all_schemes():
    """Test 1: Clean identification across all supported FEC & interleaving schemes."""
    print("=== Test 1: Noiseless Blind FEC Identification (All Schemes) ===")
    np.random.seed(42)

    info_512 = np.random.randint(0, 2, 512, dtype=np.uint8)
    coded_conv = conv_encode(info_512, add_tail=True)

    concat_payload = concatenated_encode(np.random.randint(0, 2, 48 * 8, dtype=np.uint8), n=64, k=48)

    test_cases = [
        ("Viterbi + Block (8x32)",       block_interleave(coded_conv, rows=8, cols=32), "viterbi_block", None),
        ("Viterbi + Conv (8x4)",         conv_interleave(coded_conv, depth=8, span=4, flush=True), "viterbi_conv", None),
        ("Viterbi + Pseudo-Rand Diag",   diagonal_interleave(coded_conv, block_size=256, seed=42), "viterbi_diagonal", None),
        ("Viterbi + True Diagonal",      true_diagonal_interleave(coded_conv, rows=16, cols=16), "viterbi_true_diagonal", None),
        ("Reed-Solomon RS(32,24)",       np.unpackbits(rs_encode(np.random.randint(0, 256, 24, dtype=np.uint8), 32, 24)), "rs", (32, 24)),
        ("Reed-Solomon RS(64,48)",       np.unpackbits(rs_encode(np.random.randint(0, 256, 48, dtype=np.uint8), 64, 48)), "rs", (64, 48)),
        ("Reed-Solomon RS(128,112)",     np.unpackbits(rs_encode(np.random.randint(0, 256, 112, dtype=np.uint8), 128, 112)), "rs", (128, 112)),
        ("LDPC (128,64)",                ldpc_encode(np.random.randint(0, 2, 64, dtype=np.uint8)), "ldpc", None),
        ("Concatenated (RS+Viterbi)",     concat_payload, "concatenated", (64, 48)),
    ]

    for expected_name, payload, exp_key, exp_rs_size in test_cases:
        results = identify_fec_scheme(payload)
        assert len(results) > 0, f"No scheme detected for {expected_name}"
        top = results[0]
        runner_up_str = f"{results[1].name} ({results[1].confidence*100:.1f}%)" if len(results) > 1 else "None"

        print(f"  Target: {expected_name:28s} -> Detected: {top.name:28s} [Conf: {top.confidence*100:5.1f}% | Runner-up: {runner_up_str}]")
        assert top.scheme_key == exp_key, f"Scheme key mismatch for {expected_name}: got {top.scheme_key}, expected {exp_key}"
        if exp_rs_size is not None:
            assert top.rs_size == exp_rs_size, f"RS size mismatch for {expected_name}: got {top.rs_size}, expected {exp_rs_size}"
        assert top.confidence >= 0.85, f"Confidence too low for clean {expected_name}: {top.confidence}"

    print("  [PASS] All noiseless schemes blindly identified with high confidence!\n")


def test_noisy_rf_pipeline_schemes():
    """Test 2: Blind identification under realistic RF channel conditions (SNR 15dB, CFO 300Hz, Phase Inversion)."""
    print("=== Test 2: Noisy Channel Blind FEC Identification (RF Pipeline) ===")
    np.random.seed(101)
    fs = 1_000_000
    sps = 8
    snr_db = 15.0
    cfo_hz = 300.0
    phase_rad = 3.14159

    # Case A: Noisy RS(64,48)
    rs_bytes = np.random.randint(0, 256, 48, dtype=np.uint8)
    rs_cw = rs_encode(rs_bytes, n=64, k=48)
    rs_payload = np.unpackbits(rs_cw)
    rs_frame = np.concatenate([DEFAULT_SYNC_32, rs_payload])

    symbols = 2 * rs_frame - 1
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    h = _rrc_filter(beta=0.35, span=8, sps=sps)
    tx = np.convolve(up, h, mode="same")
    rx = _add_awgn(_apply_offsets(tx, fs, cfo_hz, phase_rad), snr_db)
    rx_bits = demodulate(rx, fs, "BPSK", sps=sps)
    sync_res = find_sync(rx_bits, sync_pattern=DEFAULT_SYNC_32, modulation_type="BPSK")
    rx_payload = sync_res["payload_bits"][:len(rs_payload)]

    res_rs = identify_fec_scheme(rx_payload)
    assert len(res_rs) > 0
    top_rs = res_rs[0]
    print(f"  [Noisy RS(64,48)]  -> Detected: {top_rs.name} (Conf: {top_rs.confidence*100:.1f}%, {top_rs.details})")
    assert top_rs.scheme_key == "rs" and top_rs.rs_size == (64, 48)

    # Case B: Noisy LDPC(128,64)
    ldpc_info = np.random.randint(0, 2, 64, dtype=np.uint8)
    ldpc_cw = ldpc_encode(ldpc_info)
    ldpc_frame = np.concatenate([DEFAULT_SYNC_32, ldpc_cw])

    symbols = 2 * ldpc_frame - 1
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    tx = np.convolve(up, h, mode="same")
    rx = _add_awgn(_apply_offsets(tx, fs, cfo_hz, phase_rad), snr_db)
    rx_bits = demodulate(rx, fs, "BPSK", sps=sps)
    sync_res = find_sync(rx_bits, sync_pattern=DEFAULT_SYNC_32, modulation_type="BPSK")
    rx_payload = sync_res["payload_bits"][:len(ldpc_cw)]

    res_ldpc = identify_fec_scheme(rx_payload)
    assert len(res_ldpc) > 0
    top_ldpc = res_ldpc[0]
    print(f"  [Noisy LDPC]       -> Detected: {top_ldpc.name} (Conf: {top_ldpc.confidence*100:.1f}%, {top_ldpc.details})")
    assert top_ldpc.scheme_key == "ldpc"

    # Case C: Noisy Viterbi + Block
    v_info = np.random.randint(0, 2, 256, dtype=np.uint8)
    v_coded = conv_encode(v_info, add_tail=True)
    v_interleaved = block_interleave(v_coded, rows=8, cols=32)
    v_frame = np.concatenate([DEFAULT_SYNC_32, v_interleaved])

    symbols = 2 * v_frame - 1
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    tx = np.convolve(up, h, mode="same")
    rx = _add_awgn(_apply_offsets(tx, fs, cfo_hz, phase_rad), snr_db)
    rx_bits = demodulate(rx, fs, "BPSK", sps=sps)
    sync_res = find_sync(rx_bits, sync_pattern=DEFAULT_SYNC_32, modulation_type="BPSK")
    rx_payload = sync_res["payload_bits"][:len(v_interleaved)]

    res_v = identify_fec_scheme(rx_payload)
    assert len(res_v) > 0
    top_v = res_v[0]
    print(f"  [Noisy Viterbi+Block] -> Detected: {top_v.name} (Conf: {top_v.confidence*100:.1f}%, {top_v.details})")
    assert top_v.scheme_key == "viterbi_block"

    # Case D: Noisy Viterbi + Conv (8x4)
    vc_info = np.random.randint(0, 2, 256, dtype=np.uint8)
    vc_coded = conv_encode(vc_info, add_tail=True)
    vc_interleaved = conv_interleave(vc_coded, depth=8, span=4, flush=True)
    vc_frame = np.concatenate([DEFAULT_SYNC_32, vc_interleaved])

    symbols = 2 * vc_frame - 1
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    tx = np.convolve(up, h, mode="same")
    rx = _add_awgn(_apply_offsets(tx, fs, cfo_hz, phase_rad), snr_db)
    rx_bits = demodulate(rx, fs, "BPSK", sps=sps)
    sync_res = find_sync(rx_bits, sync_pattern=DEFAULT_SYNC_32, modulation_type="BPSK")
    rx_payload = sync_res["payload_bits"][:len(vc_interleaved)]

    res_vc = identify_fec_scheme(rx_payload)
    assert len(res_vc) > 0
    top_vc = res_vc[0]
    print(f"  [Noisy Viterbi+Conv]  -> Detected: {top_vc.name} (Conf: {top_vc.confidence*100:.1f}%, {top_vc.details})")
    assert top_vc.scheme_key == "viterbi_conv"

    # Case E: Noisy Viterbi + Diagonal
    vd_info = np.random.randint(0, 2, 256, dtype=np.uint8)
    vd_coded = conv_encode(vd_info, add_tail=True)
    vd_interleaved = diagonal_interleave(vd_coded, block_size=256, seed=42)
    vd_frame = np.concatenate([DEFAULT_SYNC_32, vd_interleaved])

    symbols = 2 * vd_frame - 1
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    tx = np.convolve(up, h, mode="same")
    rx = _add_awgn(_apply_offsets(tx, fs, cfo_hz, phase_rad), snr_db)
    rx_bits = demodulate(rx, fs, "BPSK", sps=sps)
    sync_res = find_sync(rx_bits, sync_pattern=DEFAULT_SYNC_32, modulation_type="BPSK")
    rx_payload = sync_res["payload_bits"][:len(vd_interleaved)]

    res_vd = identify_fec_scheme(rx_payload)
    assert len(res_vd) > 0
    top_vd = res_vd[0]
    print(f"  [Noisy Viterbi+Diag]  -> Detected: {top_vd.name} (Conf: {top_vd.confidence*100:.1f}%, {top_vd.details})")
    assert top_vd.scheme_key == "viterbi_diagonal"

    # Case F: Noisy Viterbi + True Diagonal (16x16)
    vtd_info = np.random.randint(0, 2, 256, dtype=np.uint8)
    vtd_coded = conv_encode(vtd_info, add_tail=True)
    vtd_interleaved = true_diagonal_interleave(vtd_coded, rows=16, cols=16)
    vtd_frame = np.concatenate([DEFAULT_SYNC_32, vtd_interleaved])

    symbols = 2 * vtd_frame - 1
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    tx = np.convolve(up, h, mode="same")
    rx = _add_awgn(_apply_offsets(tx, fs, cfo_hz, phase_rad), snr_db)
    rx_bits = demodulate(rx, fs, "BPSK", sps=sps)
    sync_res = find_sync(rx_bits, sync_pattern=DEFAULT_SYNC_32, modulation_type="BPSK")
    rx_payload = sync_res["payload_bits"][:len(vtd_interleaved)]

    res_vtd = identify_fec_scheme(rx_payload)
    assert len(res_vtd) > 0
    top_vtd = res_vtd[0]
    print(f"  [Noisy Viterbi+TrueDiag] -> Detected: {top_vtd.name} (Conf: {top_vtd.confidence*100:.1f}%, {top_vtd.details})")
    assert top_vtd.scheme_key == "viterbi_true_diagonal"

    print("  [PASS] All noisy RF pipeline cases blindly identified with correct scheme!\n")


def test_diagonal_cross_discrimination():
    """Test 3: Verify strict discrimination between True Diagonal and Pseudo-Random Diagonal."""
    print("=== Test 3: True Diagonal vs Pseudo-Random Diagonal Cross-Discrimination ===")
    np.random.seed(77)

    info = np.random.randint(0, 2, 512, dtype=np.uint8)
    coded = conv_encode(info, add_tail=True)

    # 1. Payload with True Diagonal
    true_diag_payload = true_diagonal_interleave(coded, rows=16, cols=16)
    res_td = identify_fec_scheme(true_diag_payload)
    assert len(res_td) > 0
    top_td = res_td[0]
    print(f"  Input: True Diagonal payload        -> Top: {top_td.name} (Conf: {top_td.confidence*100:.1f}%)")
    assert top_td.scheme_key == "viterbi_true_diagonal", f"Failed: detected {top_td.scheme_key}"
    # Verify pseudo-random diagonal is not falsely declared as top
    pr_matches = [r for r in res_td if r.scheme_key == "viterbi_diagonal"]
    if pr_matches:
        print(f"    (Pseudo-random diagonal runner-up conf: {pr_matches[0].confidence*100:.1f}%)")
        assert top_td.confidence > pr_matches[0].confidence

    # 2. Payload with Pseudo-Random Diagonal
    pseudo_rand_payload = diagonal_interleave(coded, block_size=256, seed=42)
    res_pr = identify_fec_scheme(pseudo_rand_payload)
    assert len(res_pr) > 0
    top_pr = res_pr[0]
    print(f"  Input: Pseudo-Random Diag payload   -> Top: {top_pr.name} (Conf: {top_pr.confidence*100:.1f}%)")
    assert top_pr.scheme_key == "viterbi_diagonal", f"Failed: detected {top_pr.scheme_key}"
    # Verify true diagonal is not falsely declared as top
    td_matches = [r for r in res_pr if r.scheme_key == "viterbi_true_diagonal"]
    if td_matches:
        print(f"    (True diagonal runner-up conf: {td_matches[0].confidence*100:.1f}%)")
        assert top_pr.confidence > td_matches[0].confidence

    print("  [PASS] True Diagonal and Pseudo-Random Diagonal are perfectly distinguished!\n")


def test_short_and_empty_payloads():
    """Test 4: Graceful handling of short or empty payloads without crashing."""
    print("=== Test 4: Short & Empty Payload Handling ===")
    res_empty = identify_fec_scheme([])
    print(f"  Empty payload results count: {len(res_empty)}")
    assert len(res_empty) == 0

    res_short = identify_fec_scheme([1, 0, 1, 1, 0, 1])
    print(f"  Short payload (6 bits) results count: {len(res_short)}")
    assert len(res_short) == 0

    print("  [PASS] Short and empty payloads handled gracefully without errors!\n")


if __name__ == "__main__":
    print("=================================================================")
    print("     RUNNING BLIND FEC IDENTIFICATION VERIFICATION SUITE         ")
    print("=================================================================\n")
    test_noiseless_all_schemes()
    test_noisy_rf_pipeline_schemes()
    test_diagonal_cross_discrimination()
    test_short_and_empty_payloads()
    print("=================================================================")
    print("        ALL BLIND FEC IDENTIFICATION TESTS PASSED!               ")
    print("=================================================================")
