"""
test_demod.py
==============
Standalone verification script for modulation.demodulator.
Tests BPSK and QPSK bit error rate (BER) against known ground truth bits,
accounting for carrier phase ambiguity (inversion / 90-degree rotations).
"""

import numpy as np
from modulation.signal_gen import _rrc_filter, _apply_offsets, _add_awgn
from modulation.demodulator import demodulate


def test_bpsk(num_symbols=2000, sps=8, fs=1_000_000, snr_db=18.0, freq_offset_hz=250.0, phase_offset_rad=0.7):
    np.random.seed(123)
    # Generate ground-truth bits
    orig_bits = np.random.randint(0, 2, num_symbols)
    symbols = 2 * orig_bits - 1  # ±1

    upsampled = np.zeros(num_symbols * sps, dtype=complex)
    upsampled[::sps] = symbols
    h = _rrc_filter(0.35, 8, sps)
    sig = np.convolve(upsampled, h, mode="same")
    sig = _apply_offsets(sig, fs, freq_offset_hz, phase_offset_rad)
    sig = _add_awgn(sig, snr_db)

    # Demodulate
    rec_bits = demodulate(sig, fs, "BPSK", sps=sps)

    # Compare lengths
    n = min(len(orig_bits), len(rec_bits))
    orig = orig_bits[:n]
    rec = rec_bits[:n]

    # Check direct and inverted
    ber_direct = np.mean(orig != rec)
    ber_inverted = np.mean((1 - orig) != rec)
    min_ber = min(ber_direct, ber_inverted)
    locked_state = "Direct (0 deg)" if ber_direct <= ber_inverted else "Inverted (180 deg)"

    print(f"[BPSK] Symbols: {num_symbols:,} | Total Bits: {n:,} | SNR: {snr_db} dB | CFO: {freq_offset_hz} Hz")
    print(f"       Direct BER: {ber_direct*100:.2f}% | Inverted BER: {ber_inverted*100:.2f}%")
    print(f"       Best BER:   {min_ber*100:.2f}%  ({locked_state})")
    return min_ber


def test_qpsk(num_symbols=2000, sps=8, fs=1_000_000, snr_db=18.0, freq_offset_hz=250.0, phase_offset_rad=0.7):
    np.random.seed(123)
    # Generate ground-truth bits
    bits_i_raw = np.random.randint(0, 2, num_symbols)
    bits_q_raw = np.random.randint(0, 2, num_symbols)

    symbols = ((2 * bits_i_raw - 1) + 1j * (2 * bits_q_raw - 1)) / np.sqrt(2)

    upsampled = np.zeros(num_symbols * sps, dtype=complex)
    upsampled[::sps] = symbols
    h = _rrc_filter(0.35, 8, sps)
    sig = np.convolve(upsampled, h, mode="same")
    sig = _apply_offsets(sig, fs, freq_offset_hz, phase_offset_rad)
    sig = _add_awgn(sig, snr_db)

    # Demodulate
    rec_bits = demodulate(sig, fs, "QPSK", sps=sps)

    n_sym = min(num_symbols, len(rec_bits) // 2)
    b_i_orig = bits_i_raw[:n_sym]
    b_q_orig = bits_q_raw[:n_sym]

    rec_i = rec_bits[: 2 * n_sym : 2]
    rec_q = rec_bits[1 : 2 * n_sym : 2]

    # 4 rotational symmetries (0, 90, 180, 270 deg)
    symmetries = {
        "0 deg (I, Q)": (b_i_orig, b_q_orig),
        "90 deg (-Q, I)": (1 - b_q_orig, b_i_orig),
        "180 deg (-I, -Q)": (1 - b_i_orig, 1 - b_q_orig),
        "270 deg (Q, -I)": (b_q_orig, 1 - b_i_orig),
    }

    print(f"\n[QPSK] Symbols: {num_symbols:,} | Total Bits: {2*n_sym:,} | SNR: {snr_db} dB | CFO: {freq_offset_hz} Hz")
    best_ber = 1.0
    best_sym = ""
    for name, (exp_i, exp_q) in symmetries.items():
        ber_i = np.mean(exp_i != rec_i)
        ber_q = np.mean(exp_q != rec_q)
        ber_total = 0.5 * (ber_i + ber_q)
        print(f"       Rotation {name:<18}: BER = {ber_total*100:.2f}%")
        if ber_total < best_ber:
            best_ber = ber_total
            best_sym = name

    print(f"       Best BER:   {best_ber*100:.2f}%  ({best_sym})")
    return best_ber


def test_16qam(num_symbols=2000, sps=8, fs=1_000_000, snr_db=15.0, freq_offset_hz=300.0, phase_offset_rad=0.7):
    np.random.seed(123)
    # Generate ground-truth bits (4 bits per symbol: [b_I0, b_Q0, b_I1, b_Q1])
    orig_bits = np.random.randint(0, 2, num_symbols * 4)
    from modulation.signal_gen import generate_16qam_from_bits

    sig = generate_16qam_from_bits(
        orig_bits, sps=sps, sample_rate=fs, snr_db=snr_db,
        freq_offset_hz=freq_offset_hz, phase_offset_rad=phase_offset_rad
    )

    # Demodulate
    rec_bits = demodulate(sig, fs, "16QAM", sps=sps)

    n_sym = min(num_symbols, len(rec_bits) // 4)
    orig = orig_bits[: 4 * n_sym]
    rec = rec_bits[: 4 * n_sym]

    b_i0_orig = orig[0::4]
    b_q0_orig = orig[1::4]
    b_i1_orig = orig[2::4]
    b_q1_orig = orig[3::4]

    # 4 rotational symmetries for Gray-coded 16QAM:
    # 0 deg:   ( I0,  Q0, I1, Q1) -> (b_i0, b_q0, b_i1, b_q1)
    # 90 deg:  (-Q0,  I0, Q1, I1) -> (1-b_q0, b_i0, b_q1, b_i1)
    # 180 deg: (-I0, -Q0, I1, Q1) -> (1-b_i0, 1-b_q0, b_i1, b_q1)
    # 270 deg: ( Q0, -I0, Q1, I1) -> (b_q0, 1-b_i0, b_q1, b_i1)
    symmetries = {
        "0 deg": (b_i0_orig, b_q0_orig, b_i1_orig, b_q1_orig),
        "90 deg": (1 - b_q0_orig, b_i0_orig, b_q1_orig, b_i1_orig),
        "180 deg": (1 - b_i0_orig, 1 - b_q0_orig, b_i1_orig, b_q1_orig),
        "270 deg": (b_q0_orig, 1 - b_i0_orig, b_q1_orig, b_i1_orig),
    }

    print(f"\n[16QAM] Symbols: {num_symbols:,} | Total Bits: {4*n_sym:,} | SNR: {snr_db} dB | CFO: {freq_offset_hz} Hz")
    best_ber = 1.0
    best_sym = ""
    for name, (exp_i0, exp_q0, exp_i1, exp_q1) in symmetries.items():
        exp_bits = np.empty(4 * n_sym, dtype=int)
        exp_bits[0::4] = exp_i0
        exp_bits[1::4] = exp_q0
        exp_bits[2::4] = exp_i1
        exp_bits[3::4] = exp_q1
        ber = float(np.mean(exp_bits != rec))
        print(f"        Rotation {name:<18}: BER = {ber*100:.2f}%")
        if ber < best_ber:
            best_ber = ber
            best_sym = name

    print(f"        Best BER:   {best_ber*100:.2f}%  ({best_sym})")
    assert best_ber < 0.05, f"16QAM BER too high: {best_ber*100:.2f}%"
    return best_ber


def test_2fsk(num_symbols=2000, sps=8, fs=1_000_000, snr_db=15.0, freq_offset_hz=300.0, phase_offset_rad=0.7, deviation_hz=25000):
    np.random.seed(123)
    # Generate ground-truth bits
    orig_bits = np.random.randint(0, 2, num_symbols)
    from modulation.signal_gen import generate_2fsk_from_bits

    sig = generate_2fsk_from_bits(
        orig_bits, sps=sps, sample_rate=fs, snr_db=snr_db,
        freq_offset_hz=freq_offset_hz, phase_offset_rad=phase_offset_rad,
        deviation_hz=deviation_hz
    )

    # Demodulate
    rec_bits = demodulate(sig, fs, "2FSK", sps=sps, deviation_hz=deviation_hz)

    n = min(len(orig_bits), len(rec_bits))
    orig = orig_bits[:n]
    rec = rec_bits[:n]

    ber_direct = float(np.mean(orig != rec))
    ber_inverted = float(np.mean((1 - orig) != rec))
    min_ber = min(ber_direct, ber_inverted)
    locked_state = "Direct (0 deg)" if ber_direct <= ber_inverted else "Inverted (180 deg)"

    print(f"\n[2FSK] Symbols: {num_symbols:,} | Total Bits: {n:,} | SNR: {snr_db} dB | CFO: {freq_offset_hz} Hz | Dev: {deviation_hz/1e3:.0f} kHz")
    print(f"       Direct BER:   {ber_direct*100:.2f}% | Inverted BER: {ber_inverted*100:.2f}%")
    print(f"       Best BER:     {min_ber*100:.2f}%  ({locked_state})")
    assert min_ber < 0.05, f"2FSK BER too high: {min_ber*100:.2f}%"
    return min_ber


if __name__ == "__main__":
    print("=================================================================")
    print("       RUNNING DEMODULATOR SELF-TEST (BPSK/QPSK/16QAM/2FSK)      ")
    print("=================================================================\n")
    bpsk_ber = test_bpsk()
    qpsk_ber = test_qpsk()
    qam_ber  = test_16qam()
    fsk_ber  = test_2fsk()

    print("\n" + "=" * 65)
    print(f"Summary: BPSK BER  = {bpsk_ber*100:.2f}%")
    print(f"         QPSK BER  = {qpsk_ber*100:.2f}%")
    print(f"         16QAM BER = {qam_ber*100:.2f}%")
    print(f"         2FSK BER  = {fsk_ber*100:.2f}%")
    print("=================================================================")
