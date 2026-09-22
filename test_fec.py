"""
test_fec.py
===========
Full end-to-end pipeline verification for:
1. Convolutional Encoding (K=7, Rate 1/2) + Hard-decision Viterbi Decoding
2. (8 x 32) Block Interleaving / De-interleaving with burst error dispersion
3. Complete RF chain: Info Bits -> FEC -> Interleave -> Preamble -> Modulate (AWGN/CFO/Phase)
   -> Demodulate -> Sync Ambiguity Resolution -> De-interleave -> Viterbi Decode
"""

import numpy as np
from modulation.fec import conv_encode, viterbi_decode
from modulation.deinterleaver import (
    block_interleave,
    block_deinterleave,
    conv_interleave,
    conv_deinterleave,
    diagonal_interleave,
    diagonal_deinterleave,
    true_diagonal_interleave,
    true_diagonal_deinterleave,
)
from modulation.correlator import find_sync, DEFAULT_SYNC_32
from modulation.demodulator import demodulate
from modulation.signal_gen import _rrc_filter, _apply_offsets, _add_awgn


def test_fec_burst_resilience():
    """Unit test: Verify that interleaver + Viterbi corrects an 8-bit burst error."""
    np.random.seed(99)
    num_info = 512
    info_bits = np.random.randint(0, 2, num_info)

    # Encode (512 + 6 tail -> 1036 coded bits)
    coded = conv_encode(info_bits)
    # Interleave (1036 -> 1280 bits in 5 blocks of 256)
    interleaved = block_interleave(coded, rows=8, cols=32)

    # Inject an 8-bit consecutive burst error in the channel
    corrupted = interleaved.copy()
    corrupted[100:108] ^= 1  # Invert 8 consecutive channel bits

    # De-interleave
    deinterleaved = block_deinterleave(corrupted, rows=8, cols=32, original_len=len(coded))

    # Decode
    decoded = viterbi_decode(deinterleaved, num_info_bits=num_info)
    ber = np.mean(info_bits != decoded)

    print(f"[FEC & Interleaver Unit Test] Injected 8-bit burst error in channel.")
    print(f"                             Raw channel error rate: {8 / len(interleaved) * 100:.2f}%")
    print(f"                             Decoded BER:           {ber * 100:.2f}%")
    assert ber == 0.0, f"Viterbi failed to correct burst: BER = {ber}"
    print("                             [PASS] 8-bit burst error corrected with 0.00% BER!\n")


def test_full_pipeline_bpsk(num_info_bits=512, sps=8, fs=1_000_000, snr_db=15.0, cfo_hz=300.0, phase_rad=3.14159):
    """Full pipeline test with BPSK, CFO, and 180-degree phase inversion."""
    np.random.seed(42)
    info_bits = np.random.randint(0, 2, num_info_bits)

    # 1. FEC Encode: 512 info + 6 tail -> 1036 coded bits
    coded = conv_encode(info_bits)

    # 2. Interleave: 1036 -> 1280 bits (5 blocks of 256)
    interleaved = block_interleave(coded, rows=8, cols=32)
    assert len(interleaved) == 1280

    # 3. Packetize: [32-bit SYNC] + [1280 interleaved bits]
    sync = DEFAULT_SYNC_32
    packet_bits = np.concatenate([sync, interleaved])

    # 4. Modulate BPSK
    symbols = 2 * packet_bits - 1
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols
    h = _rrc_filter(0.35, 8, sps)
    sig = np.convolve(upsampled, h, mode="same")
    sig = _apply_offsets(sig, fs, cfo_hz, phase_rad)
    sig = _add_awgn(sig, snr_db)

    # 5. Demodulate
    rec_bits = demodulate(sig, fs, "BPSK", sps=sps)

    # 6. Sync & Phase Ambiguity Resolution
    sync_res = find_sync(rec_bits, sync_pattern=sync, modulation_type="BPSK")
    payload_rec = sync_res["payload_bits"][: len(interleaved)]

    # 7. De-interleave
    deinterleaved = block_deinterleave(payload_rec, rows=8, cols=32, original_len=len(coded))

    # 8. Viterbi Decode
    decoded_info = viterbi_decode(deinterleaved, num_info_bits=num_info_bits)

    # 9. Verify exact BER on the 512 info bits
    ber = np.mean(info_bits != decoded_info)

    print(f"[Full Pipeline BPSK] Info Bits: {num_info_bits} | Coded/Interleaved: {len(interleaved)} | SNR: {snr_db} dB | CFO: {cfo_hz} Hz")
    print(f"                    Sync Match: {sync_res['score']*100:.1f}% ({sync_res['variant']}) | Offset: {sync_res['offset']}")
    print(f"                    Decoded Bits: {len(decoded_info)} | Final Info BER: {ber*100:.2f}%")
    assert sync_res["offset"] == 0
    assert sync_res["score"] >= 0.95
    assert ber == 0.0, f"Pipeline BPSK BER non-zero: {ber}"
    print("                    [PASS] End-to-end BPSK recovered with 0.00% BER!\n")


def test_full_pipeline_qpsk(num_info_bits=512, sps=8, fs=1_000_000, snr_db=15.0, cfo_hz=300.0, phase_rad=1.57079):
    """Full pipeline test with QPSK, CFO, and 90-degree quadrant rotation."""
    np.random.seed(42)
    info_bits = np.random.randint(0, 2, num_info_bits)

    # 1. FEC Encode: 512 info + 6 tail -> 1036 coded bits
    coded = conv_encode(info_bits)

    # 2. Interleave: 1036 -> 1280 bits (5 blocks of 256)
    interleaved = block_interleave(coded, rows=8, cols=32)

    # 3. Packetize: [32-bit SYNC] + [1280 interleaved bits]
    sync = DEFAULT_SYNC_32
    packet_bits = np.concatenate([sync, interleaved])

    # 4. Modulate QPSK
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

    # 5. Demodulate
    rec_bits = demodulate(sig, fs, "QPSK", sps=sps)

    # 6. Sync & Phase Ambiguity Resolution
    sync_res = find_sync(rec_bits, sync_pattern=sync, modulation_type="QPSK")
    payload_rec = sync_res["payload_bits"][: len(interleaved)]

    # 7. De-interleave
    deinterleaved = block_deinterleave(payload_rec, rows=8, cols=32, original_len=len(coded))

    # 8. Viterbi Decode
    decoded_info = viterbi_decode(deinterleaved, num_info_bits=num_info_bits)

    # 9. Verify exact BER on the 512 info bits
    ber = np.mean(info_bits != decoded_info)

    print(f"[Full Pipeline QPSK] Info Bits: {num_info_bits} | Coded/Interleaved: {len(interleaved)} | SNR: {snr_db} dB | CFO: {cfo_hz} Hz")
    print(f"                    Sync Match: {sync_res['score']*100:.1f}% ({sync_res['variant']}) | Offset: {sync_res['offset']}")
    print(f"                    Decoded Bits: {len(decoded_info)} | Final Info BER: {ber*100:.2f}%")
    assert sync_res["offset"] == 0
    assert sync_res["score"] >= 0.95
    assert ber == 0.0, f"Pipeline QPSK BER non-zero: {ber}"
    print("                    [PASS] End-to-end QPSK recovered with 0.00% BER!\n")


def test_conv_interleaver_burst_resilience():
    """Unit test: Verify that (8, 4) Forney convolutional interleaver + Viterbi corrects an 8-bit burst error."""
    np.random.seed(99)
    num_info = 512
    info_bits = np.random.randint(0, 2, num_info)

    # Encode (512 + 6 tail -> 1036 coded bits)
    coded = conv_encode(info_bits)
    # Forney Convolutional Interleave (depth=8, span=4 -> total latency D=224 bits)
    interleaved = conv_interleave(coded, depth=8, span=4)

    # Inject an 8-bit consecutive burst error in the channel
    corrupted = interleaved.copy()
    corrupted[100:108] ^= 1  # Invert 8 consecutive channel bits

    # Complementary Forney Convolutional De-interleave
    deinterleaved = conv_deinterleave(corrupted, depth=8, span=4, trim_delay=True, original_len=len(coded))

    # Decode
    decoded = viterbi_decode(deinterleaved, num_info_bits=num_info)
    ber = np.mean(info_bits != decoded)

    print(f"[Conv Interleaver (8x4) Unit Test] Injected 8-bit burst error in channel.")
    print(f"                                   Raw channel error rate: {8 / len(interleaved) * 100:.2f}%")
    print(f"                                   Decoded BER:           {ber * 100:.2f}%")
    assert ber == 0.0, f"Viterbi failed to correct burst: BER = {ber}"
    print("                                   [PASS] 8-bit burst error corrected with 0.00% BER!\n")


def test_full_pipeline_conv_bpsk(num_info_bits=512, sps=8, fs=1_000_000, snr_db=15.0, cfo_hz=300.0, phase_rad=3.14159):
    """Full RF chain using Convolutional Interleaving with BPSK, CFO, and 180-deg phase inversion."""
    np.random.seed(42)
    info_bits = np.random.randint(0, 2, num_info_bits)

    # 1. FEC Encode: 512 info + 6 tail -> 1036 coded bits
    coded = conv_encode(info_bits)

    # 2. Conv Interleave: depth=8, span=4 (1036 + 224 flush -> 1260 bits)
    interleaved = conv_interleave(coded, depth=8, span=4)
    assert len(interleaved) == 1260

    # 3. Packetize: [32-bit SYNC] + [1260 conv-interleaved bits]
    sync = DEFAULT_SYNC_32
    packet_bits = np.concatenate([sync, interleaved])

    # 4. Modulate BPSK
    symbols = 2 * packet_bits - 1
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols
    h = _rrc_filter(0.35, 8, sps)
    sig = np.convolve(upsampled, h, mode="same")
    sig = _apply_offsets(sig, fs, cfo_hz, phase_rad)
    sig = _add_awgn(sig, snr_db)

    # 5. Demodulate
    rec_bits = demodulate(sig, fs, "BPSK", sps=sps)

    # 6. Sync & Phase Ambiguity Resolution
    sync_res = find_sync(rec_bits, sync_pattern=sync, modulation_type="BPSK")
    payload_rec = sync_res["payload_bits"][: len(interleaved)]

    # 7. Conv De-interleave
    deinterleaved = conv_deinterleave(payload_rec, depth=8, span=4, trim_delay=True, original_len=len(coded))

    # 8. Viterbi Decode
    decoded_info = viterbi_decode(deinterleaved, num_info_bits=num_info_bits)

    # 9. Verify exact BER on the 512 info bits
    ber = np.mean(info_bits != decoded_info)

    print(f"[Full Pipeline Conv-BPSK] Info Bits: {num_info_bits} | Interleaved: {len(interleaved)} | SNR: {snr_db} dB | CFO: {cfo_hz} Hz")
    print(f"                         Sync Match: {sync_res['score']*100:.1f}% ({sync_res['variant']}) | Offset: {sync_res['offset']}")
    print(f"                         Decoded Bits: {len(decoded_info)} | Final Info BER: {ber*100:.2f}%")
    assert sync_res["offset"] == 0
    assert sync_res["score"] >= 0.95
    assert ber == 0.0, f"Pipeline Conv-BPSK BER non-zero: {ber}"
    print("                         [PASS] End-to-end Conv-BPSK recovered with 0.00% BER!\n")


def test_diagonal_interleaver_burst_resilience():
    """Unit test: Verify that diagonal (pseudo-random) interleaver + Viterbi corrects an 8-bit burst error."""
    np.random.seed(99)
    num_info = 512
    info_bits = np.random.randint(0, 2, num_info)

    # Encode (512 + 6 tail -> 1036 coded bits)
    coded = conv_encode(info_bits)
    # Diagonal Interleave (1036 -> 1280 bits in 5 blocks of 256)
    interleaved = diagonal_interleave(coded, block_size=256, seed=42)

    # Inject an 8-bit consecutive burst error in the channel
    corrupted = interleaved.copy()
    corrupted[100:108] ^= 1  # Invert 8 consecutive channel bits

    # Diagonal De-interleave
    deinterleaved = diagonal_deinterleave(corrupted, block_size=256, seed=42, original_len=len(coded))

    # Decode
    decoded = viterbi_decode(deinterleaved, num_info_bits=num_info)
    ber = np.mean(info_bits != decoded)

    print(f"[Diagonal Interleaver Unit Test] Injected 8-bit burst error in channel.")
    print(f"                                Raw channel error rate: {8 / len(interleaved) * 100:.2f}%")
    print(f"                                Decoded BER:           {ber * 100:.2f}%")
    assert ber == 0.0, f"Viterbi failed to correct burst: BER = {ber}"
    print("                                [PASS] 8-bit burst error corrected with 0.00% BER!\n")


def test_full_pipeline_diagonal_bpsk(num_info_bits=512, sps=8, fs=1_000_000, snr_db=15.0, cfo_hz=300.0, phase_rad=3.14159):
    """Full RF chain using Diagonal (pseudo-random) Interleaving with BPSK, CFO, and 180-deg phase inversion."""
    np.random.seed(42)
    info_bits = np.random.randint(0, 2, num_info_bits)

    # 1. FEC Encode: 512 info + 6 tail -> 1036 coded bits
    coded = conv_encode(info_bits)

    # 2. Diagonal Interleave: block_size=256 (1036 -> 1280 bits)
    interleaved = diagonal_interleave(coded, block_size=256, seed=42)
    assert len(interleaved) == 1280

    # 3. Packetize: [32-bit SYNC] + [1280 diagonal-interleaved bits]
    sync = DEFAULT_SYNC_32
    packet_bits = np.concatenate([sync, interleaved])

    # 4. Modulate BPSK
    symbols = 2 * packet_bits - 1
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols
    h = _rrc_filter(0.35, 8, sps)
    sig = np.convolve(upsampled, h, mode="same")
    sig = _apply_offsets(sig, fs, cfo_hz, phase_rad)
    sig = _add_awgn(sig, snr_db)

    # 5. Demodulate
    rec_bits = demodulate(sig, fs, "BPSK", sps=sps)

    # 6. Sync & Phase Ambiguity Resolution
    sync_res = find_sync(rec_bits, sync_pattern=sync, modulation_type="BPSK")
    payload_rec = sync_res["payload_bits"][: len(interleaved)]

    # 7. Diagonal De-interleave
    deinterleaved = diagonal_deinterleave(payload_rec, block_size=256, seed=42, original_len=len(coded))

    # 8. Viterbi Decode
    decoded_info = viterbi_decode(deinterleaved, num_info_bits=num_info_bits)

    # 9. Verify exact BER on the 512 info bits
    ber = np.mean(info_bits != decoded_info)

    print(f"[Full Pipeline Diagonal-BPSK] Info Bits: {num_info_bits} | Interleaved: {len(interleaved)} | SNR: {snr_db} dB | CFO: {cfo_hz} Hz")
    print(f"                             Sync Match: {sync_res['score']*100:.1f}% ({sync_res['variant']}) | Offset: {sync_res['offset']}")
    print(f"                             Decoded Bits: {len(decoded_info)} | Final Info BER: {ber*100:.2f}%")
    assert sync_res["offset"] == 0
    assert sync_res["score"] >= 0.95
    assert ber == 0.0, f"Pipeline Diagonal-BPSK BER non-zero: {ber}"
    print("                             [PASS] End-to-end Diagonal-BPSK recovered with 0.00% BER!\n")


def test_true_diagonal_interleaver_burst_resilience():
    """Unit test: Verify that True Deterministic Diagonal Interleaver + Viterbi corrects an 8-bit burst error."""
    np.random.seed(99)
    num_info = 512
    info_bits = np.random.randint(0, 2, num_info)

    # Encode (512 + 6 tail -> 1036 coded bits)
    coded = conv_encode(info_bits)
    # True Diagonal Interleave (1036 -> 1280 bits in 5 blocks of 16x16=256)
    interleaved = true_diagonal_interleave(coded, rows=16, cols=16)

    # Inject an 8-bit consecutive burst error in the channel
    corrupted = interleaved.copy()
    corrupted[100:108] ^= 1  # Invert 8 consecutive channel bits

    # True Diagonal De-interleave
    deinterleaved = true_diagonal_deinterleave(corrupted, rows=16, cols=16, original_len=len(coded))

    # Decode
    decoded = viterbi_decode(deinterleaved, num_info_bits=num_info)
    ber = np.mean(info_bits != decoded)

    print(f"[True Diagonal Interleaver Unit Test] Injected 8-bit burst error in channel.")
    print(f"                                     Raw channel error rate: {8 / len(interleaved) * 100:.2f}%")
    print(f"                                     Decoded BER:           {ber * 100:.2f}%")
    assert ber == 0.0, f"Viterbi failed to correct burst: BER = {ber}"
    print("                                     [PASS] 8-bit burst error corrected with 0.00% BER!\n")


def test_full_pipeline_true_diagonal_bpsk(num_info_bits=512, sps=8, fs=1_000_000, snr_db=15.0, cfo_hz=300.0, phase_rad=3.14159):
    """Full RF chain using True Deterministic Diagonal (16x16) Interleaving with BPSK, CFO, and 180-deg phase inversion."""
    np.random.seed(42)
    info_bits = np.random.randint(0, 2, num_info_bits)

    # 1. FEC Encode: 512 info + 6 tail -> 1036 coded bits
    coded = conv_encode(info_bits)

    # 2. True Diagonal Interleave: rows=16, cols=16 (1036 -> 1280 bits)
    interleaved = true_diagonal_interleave(coded, rows=16, cols=16)
    assert len(interleaved) == 1280

    # 3. Packetize: [32-bit SYNC] + [1280 diagonal-interleaved bits]
    sync = DEFAULT_SYNC_32
    packet_bits = np.concatenate([sync, interleaved])

    # 4. Modulate BPSK
    symbols = 2 * packet_bits - 1
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols
    h = _rrc_filter(0.35, 8, sps)
    sig = np.convolve(upsampled, h, mode="same")
    sig = _apply_offsets(sig, fs, cfo_hz, phase_rad)
    sig = _add_awgn(sig, snr_db)

    # 5. Demodulate
    rec_bits = demodulate(sig, fs, "BPSK", sps=sps)

    # 6. Sync & Phase Ambiguity Resolution
    sync_res = find_sync(rec_bits, sync_pattern=sync, modulation_type="BPSK")
    payload_rec = sync_res["payload_bits"][: len(interleaved)]

    # 7. True Diagonal De-interleave
    deinterleaved = true_diagonal_deinterleave(payload_rec, rows=16, cols=16, original_len=len(coded))

    # 8. Viterbi Decode
    decoded_info = viterbi_decode(deinterleaved, num_info_bits=num_info_bits)

    # 9. Verify exact BER on the 512 info bits
    ber = np.mean(info_bits != decoded_info)

    print(f"[Full Pipeline True-Diagonal-BPSK] Info Bits: {num_info_bits} | Interleaved: {len(interleaved)} | SNR: {snr_db} dB | CFO: {cfo_hz} Hz")
    print(f"                                  Sync Match: {sync_res['score']*100:.1f}% ({sync_res['variant']}) | Offset: {sync_res['offset']}")
    print(f"                                  Decoded Bits: {len(decoded_info)} | Final Info BER: {ber*100:.2f}%")
    assert sync_res["offset"] == 0
    assert sync_res["score"] >= 0.95
    assert ber == 0.0, f"Pipeline True-Diagonal-BPSK BER non-zero: {ber}"
    print("                                  [PASS] End-to-end True-Diagonal-BPSK recovered with 0.00% BER!\n")


if __name__ == "__main__":
    print("=== RUNNING FEC & DE-INTERLEAVING FULL PIPELINE TESTS ===\n")
    test_fec_burst_resilience()
    test_full_pipeline_bpsk()
    test_full_pipeline_qpsk()
    test_conv_interleaver_burst_resilience()
    test_full_pipeline_conv_bpsk()
    test_diagonal_interleaver_burst_resilience()
    test_full_pipeline_diagonal_bpsk()
    test_true_diagonal_interleaver_burst_resilience()
    test_full_pipeline_true_diagonal_bpsk()
    print("=== ALL FEC & DE-INTERLEAVING TESTS PASSED! ===")
