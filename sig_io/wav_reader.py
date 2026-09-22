"""
io/wav_reader.py
================
Read mono/stereo .wav files and return complex IQ samples.

For stereo WAV:  left channel = I,  right channel = Q
For mono WAV:    real-only signal (Q = 0), returned as complex

Returns
-------
dict with keys:
    samples      : np.ndarray complex128
    sample_rate  : int (Hz)
    sample_count : int
    duration     : float (seconds)
    bit_depth    : int (8, 16, 24, 32)
    channels     : int
    file_path    : str
"""

from __future__ import annotations
import numpy as np
import wave
import struct
import os


def read_wav(file_path: str) -> dict:
    """
    Load a WAV file and return metadata + complex IQ samples.

    Parameters
    ----------
    file_path : str
        Path to the .wav file.

    Returns
    -------
    dict  (see module docstring for keys)
    """
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"WAV file not found: {file_path}")

    with wave.open(file_path, "rb") as wf:
        n_channels   = wf.getnchannels()
        samp_width   = wf.getsampwidth()   # bytes per sample
        sample_rate  = wf.getframerate()
        n_frames     = wf.getnframes()
        raw          = wf.readframes(n_frames)

    bit_depth = samp_width * 8

    # --- decode raw bytes to float64 in [-1, 1] ---
    if bit_depth == 8:
        # WAV 8-bit is unsigned
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif bit_depth == 16:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        data /= 32768.0
    elif bit_depth == 24:
        # No native dtype for 24-bit; unpack manually
        total_samples = n_frames * n_channels
        data = np.empty(total_samples, dtype=np.float64)
        raw_bytes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        # little-endian 24-bit signed
        vals = (raw_bytes[:, 0].astype(np.int32)
                | (raw_bytes[:, 1].astype(np.int32) << 8)
                | (raw_bytes[:, 2].astype(np.int32) << 16))
        # sign-extend
        vals[vals >= 0x800000] -= 0x1000000
        data = vals.astype(np.float64) / 8388608.0
    elif bit_depth == 32:
        # WAV 32-bit is usually PCM int32 (not float)
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
        data /= 2147483648.0
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    # Reshape to (n_frames, n_channels)
    data = data.reshape(-1, n_channels)

    # Build complex IQ
    if n_channels == 1:
        samples = data[:, 0].astype(np.complex128)
    elif n_channels >= 2:
        samples = data[:, 0] + 1j * data[:, 1]
    else:
        raise ValueError("Zero-channel WAV?")

    sample_count = len(samples)
    duration     = sample_count / sample_rate

    return {
        "samples":      samples,
        "sample_rate":  sample_rate,
        "sample_count": sample_count,
        "duration":     duration,
        "bit_depth":    bit_depth,
        "channels":     n_channels,
        "file_path":    file_path,
    }


def print_info(info: dict) -> None:
    """Pretty-print the metadata dict returned by read_wav()."""
    print(f"File        : {os.path.basename(info['file_path'])}")
    print(f"Sample rate : {info['sample_rate']:,} Hz")
    print(f"Samples     : {info['sample_count']:,}")
    print(f"Duration    : {info['duration']:.4f} s")
    print(f"Bit depth   : {info['bit_depth']}-bit")
    print(f"Channels    : {info['channels']}")
    print(f"dtype       : {info['samples'].dtype}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python wav_reader.py <file.wav>")
        sys.exit(1)
    info = read_wav(sys.argv[1])
    print_info(info)
