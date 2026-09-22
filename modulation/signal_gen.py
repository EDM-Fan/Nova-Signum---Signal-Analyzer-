"""
Synthetic modulated signal generator.
Generates BPSK, QPSK, 2-FSK, 16-QAM complex baseband IQ signals
with configurable SNR, phase offset, and frequency offset — used
both for training the classifier and for producing test .iq files.
"""
import numpy as np


def _rrc_filter(beta, span, sps):
    """Root-raised-cosine pulse shaping filter."""
    N = span * sps
    t = np.arange(-N / 2, N / 2 + 1) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta != 0 and abs(abs(4 * beta * ti) - 1.0) < 1e-8:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))


def _add_awgn(sig, snr_db):
    sig_power = np.mean(np.abs(sig) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig)))
    return sig + noise


def _apply_offsets(sig, sample_rate, freq_offset_hz=0.0, phase_offset_rad=0.0):
    n = np.arange(len(sig))
    sig = sig * np.exp(1j * (2 * np.pi * freq_offset_hz * n / sample_rate + phase_offset_rad))
    return sig


def generate_bpsk(num_symbols, sps, sample_rate, snr_db=15, freq_offset_hz=0, phase_offset_rad=0):
    bits = np.random.randint(0, 2, num_symbols)
    symbols = 2 * bits - 1  # +-1
    upsampled = np.zeros(num_symbols * sps, dtype=complex)
    upsampled[::sps] = symbols
    h = _rrc_filter(0.35, 8, sps)
    sig = np.convolve(upsampled, h, mode='same')
    sig = _apply_offsets(sig, sample_rate, freq_offset_hz, phase_offset_rad)
    return _add_awgn(sig, snr_db)


def generate_qpsk(num_symbols, sps, sample_rate, snr_db=15, freq_offset_hz=0, phase_offset_rad=0):
    bits_i = 2 * np.random.randint(0, 2, num_symbols) - 1
    bits_q = 2 * np.random.randint(0, 2, num_symbols) - 1
    symbols = (bits_i + 1j * bits_q) / np.sqrt(2)
    upsampled = np.zeros(num_symbols * sps, dtype=complex)
    upsampled[::sps] = symbols
    h = _rrc_filter(0.35, 8, sps)
    sig = np.convolve(upsampled, h, mode='same')
    sig = _apply_offsets(sig, sample_rate, freq_offset_hz, phase_offset_rad)
    return _add_awgn(sig, snr_db)


def bits_to_16qam_symbols(bits: np.ndarray | list[int]) -> np.ndarray:
    """
    Map a bit array to Gray-coded 16QAM complex symbols.
    4 bits per symbol: [b_I0, b_Q0, b_I1, b_Q1]
      b_0: sign bit (1: positive, 0: negative)
      b_1: magnitude bit (1: inner level +/-1, 0: outer level +/-3)
    Levels:
      00 -> -3,  01 -> -1,  11 -> +1,  10 -> +3
    Average symbol power is normalized to 1.0 (divided by sqrt(10)).
    """
    bits = np.asarray(bits, dtype=int).ravel()
    n_sym = len(bits) // 4
    b_i0 = bits[0::4][:n_sym]
    b_q0 = bits[1::4][:n_sym]
    b_i1 = bits[2::4][:n_sym]
    b_q1 = bits[3::4][:n_sym]

    i_level = np.where(b_i0 == 1, np.where(b_i1 == 1, 1.0, 3.0), np.where(b_i1 == 1, -1.0, -3.0))
    q_level = np.where(b_q0 == 1, np.where(b_q1 == 1, 1.0, 3.0), np.where(b_q1 == 1, -1.0, -3.0))

    return (i_level + 1j * q_level) / np.sqrt(10.0)


def generate_16qam_from_bits(bits, sps=8, sample_rate=1_000_000, snr_db=15, freq_offset_hz=0, phase_offset_rad=0):
    """Generate 16QAM baseband IQ signal from an explicit bit sequence."""
    symbols = bits_to_16qam_symbols(bits)
    num_symbols = len(symbols)
    upsampled = np.zeros(num_symbols * sps, dtype=complex)
    upsampled[::sps] = symbols
    h = _rrc_filter(0.35, 8, sps)
    sig = np.convolve(upsampled, h, mode='same')
    sig = _apply_offsets(sig, sample_rate, freq_offset_hz, phase_offset_rad)
    return _add_awgn(sig, snr_db)


def generate_16qam(num_symbols, sps, sample_rate, snr_db=15, freq_offset_hz=0, phase_offset_rad=0):
    bits = np.random.randint(0, 2, num_symbols * 4)
    return generate_16qam_from_bits(bits, sps, sample_rate, snr_db, freq_offset_hz, phase_offset_rad)


def generate_2fsk_from_bits(bits, sps=8, sample_rate=1_000_000, snr_db=15, freq_offset_hz=0, phase_offset_rad=0, deviation_hz=25000):
    """Generate 2FSK baseband IQ signal from an explicit bit sequence."""
    bits = np.asarray(bits, dtype=int).ravel()
    freqs = np.where(bits == 0, -deviation_hz, deviation_hz)
    freq_per_sample = np.repeat(freqs, sps)
    phase = 2 * np.pi * np.cumsum(freq_per_sample) / sample_rate
    sig = np.exp(1j * phase)
    sig = _apply_offsets(sig, sample_rate, freq_offset_hz, phase_offset_rad)
    return _add_awgn(sig, snr_db)


def generate_2fsk(num_symbols, sps, sample_rate, snr_db=15, freq_offset_hz=0, phase_offset_rad=0, deviation_hz=25000):
    bits = np.random.randint(0, 2, num_symbols)
    return generate_2fsk_from_bits(bits, sps, sample_rate, snr_db, freq_offset_hz, phase_offset_rad, deviation_hz)


GENERATORS = {
    'BPSK': generate_bpsk,
    'QPSK': generate_qpsk,
    '16QAM': generate_16qam,
    '2FSK': generate_2fsk,
}


def generate_random_example(sample_rate=1_000_000, num_symbols=2000, sps=8):
    """Pick a random modulation + random SNR/offsets — used for training data."""
    mod = np.random.choice(list(GENERATORS.keys()))
    snr_db = np.random.uniform(5, 25)
    freq_offset_hz = np.random.uniform(-2000, 2000)
    phase_offset_rad = np.random.uniform(-np.pi, np.pi)
    sig = GENERATORS[mod](num_symbols, sps, sample_rate, snr_db, freq_offset_hz, phase_offset_rad)
    return sig, mod, sample_rate
