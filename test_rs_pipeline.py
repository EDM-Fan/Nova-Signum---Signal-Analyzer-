"""
test_rs_pipeline.py
===================
Full-pipeline verification for Reed-Solomon RS(64, 48) FEC code.

Mirrors the Viterbi full-pipeline test structure:
1. Generates ground-truth info bits (48 bytes = 384 bits) and saves sidecar .npy.
2. Systematically encodes with RS(64, 48) -> 64 bytes (512 coded bits).
3. Prepends standard unencoded 32-bit SYNC word (0xEB902A3C) -> 544 packet bits.
4. Modulates to baseband IQ (BPSK / QPSK, RRC pulse shaping, sps=8, Fs=1MHz).
5. Injects channel impairments: AWGN (15 dB SNR), CFO (300 Hz), phase rotation (180 deg / 90 deg).
6. Saves real float32 .iq file to samples/rs_bpsk_full_pipeline_demo.iq.
7. Reads from the .iq file via sig_io.iq_reader.
8. Executes full receiver chain: Demodulate -> Find Sync (ambiguity resolution) -> RS Decode.
9. Validates recovered information bits against ground truth and reports BER.
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
from sig_io.iq_reader import read_iq

SAMPLES_DIR = os.path.join(_ROOT, "samples")


def test_rs_pipeline_bpsk_file_backed(
    n: int = 64,
    k: int = 48,
    sample_rate: int = 1_000_000,
    sps: int = 8,
    snr_db: float = 15.0,
    freq_offset_hz: float = 300.0,
    phase_offset_rad: float = 3.14159,
    seed: int = 42,
):
    """
    Test RS(64, 48) full pipeline with real .iq and .npy file generation for BPSK.
    """
    np.random.seed(seed)

    # 1. Generate 48 random info bytes (384 info bits)
    info_bytes = np.random.randint(0, 256, k, dtype=np.uint8)
    info_bits = np.unpackbits(info_bytes)

    # 2. RS Encode: 48 bytes -> 64 bytes (512 coded bits)
    codeword_bytes = rs_encode(info_bytes, n=n, k=k)
    codeword_bits = np.unpackbits(codeword_bytes)
    assert len(codeword_bits) == n * 8

    # 3. Packetize: [32-bit SYNC] + [512 RS codeword bits]
    sync_word = DEFAULT_SYNC_32
    packet_bits = np.concatenate([sync_word, codeword_bits])

    # 4. Modulate BPSK
    symbols = 2 * packet_bits - 1
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols

    h = _rrc_filter(beta=0.35, span=8, sps=sps)
    sig = np.convolve(upsampled, h, mode="same")
    sig = _apply_offsets(sig, sample_rate, freq_offset_hz, phase_offset_rad)
    sig = _add_awgn(sig, snr_db)

    # 5. Save real float32 .iq file and .npy sidecar ground truth
    iq_path = os.path.join(SAMPLES_DIR, "rs_bpsk_full_pipeline_demo.iq")
    npy_path = os.path.join(SAMPLES_DIR, "rs_bpsk_full_pipeline_demo_bits.npy")

    raw_iq = np.empty(2 * len(sig), dtype=np.float32)
    raw_iq[0::2] = sig.real.astype(np.float32)
    raw_iq[1::2] = sig.imag.astype(np.float32)
    raw_iq.tofile(iq_path)
    np.save(npy_path, info_bits)

    print("=== [RS Pipeline Test 1: BPSK with Real .iq File Generation] ===")
    print(f"  Generated IQ File   : {os.path.basename(iq_path)} ({os.path.getsize(iq_path):,} bytes)")
    print(f"  Ground Truth Sidecar: {os.path.basename(npy_path)}")
    print(f"  Payload Size        : {k} bytes ({len(info_bits)} bits) -> Codeword: {n} bytes ({len(codeword_bits)} bits)")
    print(f"  Channel Impairments : SNR = {snr_db} dB | CFO = {freq_offset_hz} Hz | Phase Inversion = {phase_offset_rad:.2f} rad")

    # 6. Read back through IQ reader (mimicking GUI file load)
    loaded = read_iq(iq_path, sample_rate=sample_rate)
    rec_samples = loaded["samples"]

    # 7. Demodulate
    rec_bits = demodulate(rec_samples, sample_rate, "BPSK", sps=sps)

    # 8. Sync correlation and 180-degree phase ambiguity resolution
    sync_res = find_sync(rec_bits, sync_pattern=sync_word, modulation_type="BPSK")
    payload_rec_bits = sync_res["payload_bits"][: len(codeword_bits)]

    # 9. Pack bits to bytes and RS Decode
    rec_codeword_bytes = np.packbits(payload_rec_bits)
    decoded_bytes = rs_decode(rec_codeword_bytes, n=n, k=k)
    decoded_bits = np.unpackbits(decoded_bytes)

    # 10. Verify BER against ground truth
    ground_truth = np.load(npy_path)
    bit_errors = int(np.sum(ground_truth != decoded_bits))
    ber = float(bit_errors / len(ground_truth))

    print(f"  Sync Correlation    : Offset = {sync_res['offset']} | Match = {sync_res['score']*100:.1f}% ({sync_res['variant']})")
    print(f"  Decoded Payload     : {len(decoded_bytes)} bytes ({len(decoded_bits)} bits)")
    print(f"  Final Info BER      : {ber*100:.2f}% ({bit_errors} bit errors)")

    assert sync_res["offset"] == 0, f"Sync offset non-zero: {sync_res['offset']}"
    assert sync_res["score"] >= 0.95, f"Sync match score too low: {sync_res['score']}"
    assert ber == 0.0, f"RS Pipeline BPSK BER non-zero: {ber}"
    print("  [PASS] Full BPSK RF chain with Reed-Solomon FEC recovered with 0.00% BER!\n")


def test_rs_pipeline_qpsk(
    n: int = 64,
    k: int = 48,
    sample_rate: int = 1_000_000,
    sps: int = 8,
    snr_db: float = 15.0,
    freq_offset_hz: float = 300.0,
    phase_offset_rad: float = 1.57079,
    seed: int = 42,
):
    """
    Test RS(64, 48) full pipeline with QPSK modulation and 90-degree quadrant rotation.
    """
    np.random.seed(seed)

    # 1. 48 info bytes
    info_bytes = np.random.randint(0, 256, k, dtype=np.uint8)
    info_bits = np.unpackbits(info_bytes)

    # 2. RS Encode: 48 -> 64 bytes (512 coded bits)
    codeword_bytes = rs_encode(info_bytes, n=n, k=k)
    codeword_bits = np.unpackbits(codeword_bytes)

    # 3. Packetize: [32-bit SYNC] + [512 RS codeword bits]
    sync_word = DEFAULT_SYNC_32
    packet_bits = np.concatenate([sync_word, codeword_bits])

    # 4. Modulate QPSK
    num_symbols = len(packet_bits) // 2
    b_i = packet_bits[0::2]
    b_q = packet_bits[1::2]
    symbols = ((2 * b_i - 1) + 1j * (2 * b_q - 1)) / np.sqrt(2)

    upsampled = np.zeros(num_symbols * sps, dtype=complex)
    upsampled[::sps] = symbols

    h = _rrc_filter(beta=0.35, span=8, sps=sps)
    sig = np.convolve(upsampled, h, mode="same")
    sig = _apply_offsets(sig, sample_rate, freq_offset_hz, phase_offset_rad)
    sig = _add_awgn(sig, snr_db)

    print("=== [RS Pipeline Test 2: QPSK with 90-deg Quadrant Rotation] ===")
    print(f"  Payload Size        : {k} bytes ({len(info_bits)} bits) -> Codeword: {n} bytes ({len(codeword_bits)} bits)")
    print(f"  Channel Impairments : SNR = {snr_db} dB | CFO = {freq_offset_hz} Hz | Phase Rotation = {phase_offset_rad:.2f} rad")

    # 5. Demodulate QPSK
    rec_bits = demodulate(sig, sample_rate, "QPSK", sps=sps)

    # 6. Sync correlation and 90-degree quadrant ambiguity resolution
    sync_res = find_sync(rec_bits, sync_pattern=sync_word, modulation_type="QPSK")
    payload_rec_bits = sync_res["payload_bits"][: len(codeword_bits)]

    # 7. RS Decode
    rec_codeword_bytes = np.packbits(payload_rec_bits)
    decoded_bytes = rs_decode(rec_codeword_bytes, n=n, k=k)
    decoded_bits = np.unpackbits(decoded_bytes)

    # 8. BER verification
    bit_errors = int(np.sum(info_bits != decoded_bits))
    ber = float(bit_errors / len(info_bits))

    print(f"  Sync Correlation    : Offset = {sync_res['offset']} | Match = {sync_res['score']*100:.1f}% ({sync_res['variant']})")
    print(f"  Decoded Payload     : {len(decoded_bytes)} bytes ({len(decoded_bits)} bits)")
    print(f"  Final Info BER      : {ber*100:.2f}% ({bit_errors} bit errors)")

    assert sync_res["offset"] == 0, f"Sync offset non-zero: {sync_res['offset']}"
    assert sync_res["score"] >= 0.95, f"Sync match score too low: {sync_res['score']}"
    assert ber == 0.0, f"RS Pipeline QPSK BER non-zero: {ber}"
    print("  [PASS] Full QPSK RF chain with Reed-Solomon FEC recovered with 0.00% BER!\n")


if __name__ == "__main__":
    print("=================================================================")
    print("      RUNNING REED-SOLOMON FULL RF PIPELINE VERIFICATION         ")
    print("=================================================================\n")
    test_rs_pipeline_bpsk_file_backed()
    test_rs_pipeline_qpsk()
    print("=================================================================")
    print("      ALL REED-SOLOMON RF PIPELINE TESTS PASSED (0.00% BER)      ")
    print("=================================================================")
