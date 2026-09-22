"""
Generate synthetic test IQ files: BPSK, QPSK, 2-FSK.

Format: interleaved float32 (I, Q, I, Q, ...)
Filename convention: <modulation>_<samp_rate>sps_f32.iq

SNR ~15-20 dB.
"""

import numpy as np
import os

OUTPUT_DIR = os.path.dirname(__file__)
FS = 1_000_000        # 1 MHz sample rate
DURATION = 3.0        # seconds
SYMBOL_RATE = 50_000  # 50 ksym/s
SNR_DB = 18           # target SNR in dB


def _add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add complex AWGN to reach the desired SNR (in dB)."""
    sig_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise_std = np.sqrt(noise_power / 2)
    noise = noise_std * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise


def _save_iq(samples: np.ndarray, name: str) -> str:
    """Save complex samples as interleaved float32 .iq file."""
    path = os.path.join(OUTPUT_DIR, name)
    iq = np.empty(2 * len(samples), dtype=np.float32)
    iq[0::2] = samples.real.astype(np.float32)
    iq[1::2] = samples.imag.astype(np.float32)
    iq.tofile(path)
    print(f"  Saved {path}  ({len(samples)} samples, {os.path.getsize(path)/1024:.1f} KB)")
    return path


def generate_bpsk() -> str:
    """BPSK: symbols ±1, raised-cosine filtered, AWGN."""
    sps = FS // SYMBOL_RATE          # samples per symbol
    n_symbols = int(DURATION * SYMBOL_RATE)
    bits = np.random.randint(0, 2, n_symbols)
    symbols = 2 * bits - 1           # map 0->-1, 1->+1  (real-valued)
    # upsample
    upsampled = np.zeros(n_symbols * sps)
    upsampled[::sps] = symbols

    # Root-raised-cosine filter (simple approximation via sinc)
    from scipy.signal import firwin
    num_taps = 8 * sps + 1
    rrc = firwin(num_taps, 1.0 / sps, window=("kaiser", 5))
    from scipy.signal import lfilter
    filtered = lfilter(rrc, [1.0], upsampled)

    # Carrier (centred at 0 Hz — baseband IQ)
    t = np.arange(len(filtered)) / FS
    fc = 0.0                         # baseband
    carrier = np.exp(1j * 2 * np.pi * fc * t)
    iq = filtered.astype(complex) * carrier

    iq = _add_awgn(iq, SNR_DB)
    return _save_iq(iq, f"bpsk_{FS//1000}ksps_f32.iq")


def generate_qpsk() -> str:
    """QPSK: 4-phase constellation, raised-cosine filtered, AWGN."""
    sps = FS // SYMBOL_RATE
    n_symbols = int(DURATION * SYMBOL_RATE)
    dibits = np.random.randint(0, 4, n_symbols)
    # QPSK phases: 45, 135, 225, 315 degrees
    phases = np.array([np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4])
    symbols = np.exp(1j * phases[dibits])

    # Upsample & RRC filter separately for I and Q
    from scipy.signal import firwin, lfilter
    num_taps = 8 * sps + 1
    rrc = firwin(num_taps, 1.0 / sps, window=("kaiser", 5))

    up_i = np.zeros(n_symbols * sps)
    up_q = np.zeros(n_symbols * sps)
    up_i[::sps] = symbols.real
    up_q[::sps] = symbols.imag

    fi = lfilter(rrc, [1.0], up_i)
    fq = lfilter(rrc, [1.0], up_q)
    iq = fi + 1j * fq

    iq = _add_awgn(iq, SNR_DB)
    return _save_iq(iq, f"qpsk_{FS//1000}ksps_f32.iq")


def generate_2fsk() -> str:
    """2-FSK (binary FSK): deviation ±25 kHz, AWGN."""
    n_samples = int(DURATION * FS)
    sps = FS // SYMBOL_RATE
    n_symbols = n_samples // sps
    bits = np.random.randint(0, 2, n_symbols)
    deviation = 25_000               # Hz
    freqs = np.where(bits == 0, -deviation, +deviation)

    # Build instantaneous frequency array at sample rate
    inst_freq = np.repeat(freqs, sps).astype(float)[:n_samples]

    # Integrate to get phase (continuous-phase FSK)
    phase = 2 * np.pi * np.cumsum(inst_freq) / FS
    iq = np.exp(1j * phase)

    iq = _add_awgn(iq, SNR_DB)
    return _save_iq(iq, f"2fsk_{FS//1000}ksps_f32.iq")


def generate_16qam() -> str:
    """16-QAM: 16-point constellation, raised-cosine filtered, AWGN."""
    sps = FS // SYMBOL_RATE
    n_symbols = int(DURATION * SYMBOL_RATE)
    # 16-QAM alphabet for I and Q is +/-1, +/-3
    alphabet = np.array([-3, -1, 1, 3])
    i_syms = np.random.choice(alphabet, n_symbols)
    q_syms = np.random.choice(alphabet, n_symbols)
    symbols = i_syms + 1j * q_syms
    
    # Normalize power (average power of +/-1, +/-3 is 10)
    symbols = symbols / np.sqrt(10)

    from scipy.signal import firwin, lfilter
    num_taps = 8 * sps + 1
    rrc = firwin(num_taps, 1.0 / sps, window=("kaiser", 5))

    up_i = np.zeros(n_symbols * sps)
    up_q = np.zeros(n_symbols * sps)
    up_i[::sps] = symbols.real
    up_q[::sps] = symbols.imag

    fi = lfilter(rrc, [1.0], up_i)
    fq = lfilter(rrc, [1.0], up_q)
    iq = fi + 1j * fq

    iq = _add_awgn(iq, SNR_DB)
    return _save_iq(iq, f"16qam_{FS//1000}ksps_f32.iq")


if __name__ == "__main__":
    np.random.seed(42)
    print("Generating synthetic IQ test signals ...")
    print(f"  Sample rate : {FS/1e6:.1f} MHz")
    print(f"  Duration    : {DURATION} s")
    print(f"  Symbol rate : {SYMBOL_RATE/1e3:.0f} ksym/s")
    print(f"  Target SNR  : {SNR_DB} dB\n")

    generate_bpsk()
    generate_qpsk()
    generate_2fsk()
    generate_16qam()
    print("\nDone.")
