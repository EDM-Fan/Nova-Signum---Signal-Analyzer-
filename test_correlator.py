"""
test_correlator.py
==================
Full pipeline test for bit-stream correlation and phase ambiguity resolution.
Pipeline:
1. Construct packet: [LEAD-IN BITS] + [32-BIT SYNC WORD] + [RANDOM PAYLOAD BITS]
2. Modulate with BPSK / QPSK, adding AWGN, CFO, and specific phase rotations (0, 90, 180, 270 deg)
3. Demodulate IQ samples -> raw bit array
4. Run find_sync() -> detects sync offset, correlation score, and phase variant
5. Extract payload from corrected_bits and verify 0.00% Payload BER
"""

import numpy as np
from modulation.signal_gen import _rrc_filter, _apply_offsets, _add_awgn
from modulation.demodulator import demodulate
from modulation.correlator import (
    find_sync,
    parse_sync_word,
    DEFAULT_SYNC_32,
    SYNC_CCSDS_32,
    SYNC_ALT_B_32,
    SYNC_ALT_C_32,
)


def test_bpsk_sync(lead_in_bits=40, payload_bits_count=1000, sps=8, fs=1_000_000, snr_db=18.0, cfo_hz=250.0, phase_rad=3.14159, sync_pattern=None):
    """Test BPSK packet synchronization with injected phase inversion (180 deg)."""
    np.random.seed(42)
    sync = parse_sync_word(sync_pattern)
    lead_in = np.random.randint(0, 2, lead_in_bits)
    payload = np.random.randint(0, 2, payload_bits_count)
    packet_bits = np.concatenate([lead_in, sync, payload])
    expected_offset = lead_in_bits

    # Modulate BPSK
    symbols = 2 * packet_bits - 1
    num_symbols = len(symbols)
    upsampled = np.zeros(num_symbols * sps, dtype=complex)
    upsampled[::sps] = symbols
    h = _rrc_filter(0.35, 8, sps)
    sig = np.convolve(upsampled, h, mode="same")
    sig = _apply_offsets(sig, fs, cfo_hz, phase_rad)
    sig = _add_awgn(sig, snr_db)

    # Demodulate
    rec_bits = demodulate(sig, fs, "BPSK", sps=sps)

    # Sync correlation & ambiguity resolution (pass sync_pattern as provided)
    res = find_sync(rec_bits, sync_pattern=sync_pattern, modulation_type="BPSK")

    # Verify payload
    rec_payload = res["payload_bits"][: len(payload)]
    payload_ber = np.mean(payload != rec_payload)

    print(f"[BPSK SYNC] Injected Phase: {phase_rad:.2f} rad | Expected Sync Offset: {expected_offset}")
    print(f"            Detected Offset: {res['offset']} | Match Score: {res['score']*100:.1f}% | Variant: {res['variant']}")
    print(f"            Payload Bits Checked: {len(rec_payload)} | Payload BER: {payload_ber*100:.2f}%")

    assert res["offset"] == expected_offset, f"Offset mismatch: expected {expected_offset}, got {res['offset']}"
    assert res["score"] >= 0.95, f"Correlation score too low: {res['score']}"
    assert payload_ber == 0.0, f"Payload BER is not zero: {payload_ber}"
    print("            [PASS] Sync detected and payload recovered with 0.00% BER!\n")


def test_qpsk_sync(lead_in_bits=40, payload_bits_count=1000, sps=8, fs=1_000_000, snr_db=18.0, cfo_hz=250.0, phase_rad=1.57079, sync_pattern=None):
    """Test QPSK packet synchronization with injected phase rotation (90 deg)."""
    np.random.seed(42)
    sync = parse_sync_word(sync_pattern)
    lead_in = np.random.randint(0, 2, lead_in_bits)
    payload = np.random.randint(0, 2, payload_bits_count)
    packet_bits = np.concatenate([lead_in, sync, payload])
    expected_offset = lead_in_bits

    # Ensure even number of bits for QPSK symbols
    if len(packet_bits) % 2 != 0:
        packet_bits = np.append(packet_bits, 0)

    num_symbols = len(packet_bits) // 2
    b_i = packet_bits[0::2]
    b_q = packet_bits[1::2]
    symbols = ((2 * b_i - 1) + 1j * (2 * b_q - 1)) / np.sqrt(2)

    upsampled = np.zeros(num_symbols * sps, dtype=complex)
    upsampled[::sps] = symbols
    h = _rrc_filter(0.35, 8, sps)
    sig = np.convolve(upsampled, h, mode="same")
    sig = _apply_offsets(sig, fs, cfo_hz, phase_rad)
    sig = _add_awgn(sig, snr_db)

    # Demodulate
    rec_bits = demodulate(sig, fs, "QPSK", sps=sps)

    # Sync correlation & ambiguity resolution
    res = find_sync(rec_bits, sync_pattern=sync_pattern, modulation_type="QPSK")

    # Verify payload
    rec_payload = res["payload_bits"][: len(payload)]
    payload_ber = np.mean(payload != rec_payload)

    print(f"[QPSK SYNC] Injected Phase: {phase_rad:.2f} rad | Expected Sync Offset: {expected_offset}")
    print(f"            Detected Offset: {res['offset']} | Match Score: {res['score']*100:.1f}% | Variant: {res['variant']}")
    print(f"            Payload Bits Checked: {len(rec_payload)} | Payload BER: {payload_ber*100:.2f}%")

    assert res["offset"] == expected_offset, f"Offset mismatch: expected {expected_offset}, got {res['offset']}"
    assert res["score"] >= 0.95, f"Correlation score too low: {res['score']}"
    assert payload_ber == 0.0, f"Payload BER is not zero: {payload_ber}"
    print("            [PASS] Sync detected and payload recovered with 0.00% BER!\n")


def test_all_rotations():
    """Verify all 4 QPSK rotations: 0, 90, 180, 270 degrees."""
    print("=== TESTING ALL 4 QPSK QUADRANT ROTATIONS ===")
    test_rotations = [
        (0.0, "0 deg"),
        (np.pi / 2, "90 deg"),
        (np.pi, "180 deg"),
        (3 * np.pi / 2, "270 deg"),
    ]
    for angle, expected_name in test_rotations:
        test_qpsk_sync(lead_in_bits=24, payload_bits_count=500, phase_rad=angle)


if __name__ == "__main__":
    print("=== RUNNING BIT-STREAM CORRELATOR SELF-TEST ===\n")
    print("--- 1. BPSK Inverted Phase Test (Default Sync) ---")
    test_bpsk_sync(lead_in_bits=40, phase_rad=np.pi, sync_pattern=DEFAULT_SYNC_32)

    print("--- 2. BPSK Sync Test (Alt A / CCSDS: 0x1ACFFC1D) ---")
    test_bpsk_sync(lead_in_bits=40, phase_rad=np.pi, sync_pattern=SYNC_CCSDS_32)

    print("--- 3. BPSK Sync Test (Alt B Hex String: '0xFAF334BE') ---")
    test_bpsk_sync(lead_in_bits=40, phase_rad=np.pi, sync_pattern="0xFAF334BE")

    print("--- 4. BPSK Sync Test (Alt C Preset Name: 'Alt C (0x352EF853)') ---")
    test_bpsk_sync(lead_in_bits=40, phase_rad=np.pi, sync_pattern="Alt C (0x352EF853)")

    print("--- 5. QPSK Rotations Test (Default Sync) ---")
    test_all_rotations()

    print("=== ALL CORRELATOR TESTS PASSED SUCCESSFULLY! ===")

