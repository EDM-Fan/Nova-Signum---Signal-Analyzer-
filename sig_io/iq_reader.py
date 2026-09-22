"""
io/iq_reader.py
===============
Read raw binary IQ files (interleaved I/Q pairs) in multiple formats.

Supported formats (auto-detected from file extension suffix):
    _i8.iq   / _s8.iq   : int8    signed  8-bit
    _u8.iq              : uint8   unsigned 8-bit (0..255 centre=128)
    _i16.iq  / _s16.iq  : int16   signed 16-bit  little-endian
    _f32.iq             : float32 32-bit  little-endian  ← GNU Radio default
    _f64.iq             : float64 64-bit  little-endian
    (no suffix match)   : auto-detect by trying float32

Returns
-------
dict with keys:
    samples      : np.ndarray complex128
    sample_rate  : int   (Hz)  — passed in or 0 if unknown
    sample_count : int
    duration     : float (s)   — 0 if sample_rate unknown
    iq_format    : str   ('int8','uint8','int16','float32','float64')
    file_path    : str
"""

from __future__ import annotations
import numpy as np
import os
import re


# Regex patterns to sniff dtype from filename
_DTYPE_PATTERNS = [
    (re.compile(r"_f64\.iq$",        re.I), "float64"),
    (re.compile(r"_f32\.iq$",        re.I), "float32"),
    (re.compile(r"_(i16|s16)\.iq$",  re.I), "int16"),
    (re.compile(r"_(i8|s8)\.iq$",    re.I), "int8"),
    (re.compile(r"_u8\.iq$",         re.I), "uint8"),
    (re.compile(r"\.(cu8|u8)$",      re.I), "uint8"),
    (re.compile(r"\.iq$",            re.I), "float32"),   # fallback
]

_NP_DTYPE_MAP = {
    "float64": np.float64,
    "float32": np.float32,
    "int16":   np.int16,
    "int8":    np.int8,
    "uint8":   np.uint8,
}

_SCALE_MAP = {
    "float64": 1.0,
    "float32": 1.0,
    "int16":   1.0 / 32768.0,
    "int8":    1.0 / 128.0,
    "uint8":   1.0 / 128.0,   # will also subtract 128
}


def _detect_format(file_path: str) -> str:
    basename = os.path.basename(file_path)
    for pattern, fmt in _DTYPE_PATTERNS:
        if pattern.search(basename):
            return fmt
    return "float32"


def read_iq(
    file_path: str,
    sample_rate: int = 0,
    fmt: str | None = None,
    max_samples: int | None = None,
) -> dict:
    """
    Load a raw IQ file and return metadata + complex samples.

    Parameters
    ----------
    file_path   : path to the .iq file
    sample_rate : sample rate in Hz (0 if unknown)
    fmt         : force dtype ('int8','uint8','int16','float32','float64')
                  If None, auto-detected from filename.
    max_samples : if given, read at most this many complex samples
    """
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"IQ file not found: {file_path}")

    iq_format = fmt if fmt else _detect_format(file_path)
    np_dtype  = _NP_DTYPE_MAP[iq_format]
    scale     = _SCALE_MAP[iq_format]

    # Determine how many raw values to read
    file_size    = os.path.getsize(file_path)
    bytes_per_val = np_dtype().itemsize
    total_vals   = file_size // bytes_per_val
    total_cplx   = total_vals // 2           # must be even (I/Q pairs)

    if max_samples is not None:
        total_cplx = min(total_cplx, max_samples)

    n_read = total_cplx * 2                  # raw scalar values to read

    raw = np.fromfile(file_path, dtype=np_dtype, count=n_read)

    # Normalise
    raw = raw.astype(np.float64) * scale
    if iq_format == "uint8":
        raw -= 1.0                           # centre at 0  (128*scale - 1 = 0)

    # Interleave → complex
    samples = raw[0::2] + 1j * raw[1::2]

    sample_count = len(samples)
    duration     = (sample_count / sample_rate) if sample_rate else 0.0

    return {
        "samples":      samples,
        "sample_rate":  sample_rate,
        "sample_count": sample_count,
        "duration":     duration,
        "iq_format":    iq_format,
        "file_path":    file_path,
    }


def print_info(info: dict) -> None:
    """Pretty-print the metadata dict returned by read_iq()."""
    print(f"File         : {os.path.basename(info['file_path'])}")
    print(f"IQ format    : {info['iq_format']}")
    print(f"Samples      : {info['sample_count']:,}")
    if info['sample_rate']:
        print(f"Sample rate  : {info['sample_rate']:,} Hz")
        print(f"Duration     : {info['duration']:.4f} s")
    else:
        print("Sample rate  : unknown")
    print(f"dtype        : {info['samples'].dtype}")
    print(f"I range      : [{info['samples'].real.min():.4f}, {info['samples'].real.max():.4f}]")
    print(f"Q range      : [{info['samples'].imag.min():.4f}, {info['samples'].imag.max():.4f}]")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python iq_reader.py <file.iq> [sample_rate]")
        sys.exit(1)
    fs = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    info = read_iq(sys.argv[1], sample_rate=fs)
    print_info(info)
