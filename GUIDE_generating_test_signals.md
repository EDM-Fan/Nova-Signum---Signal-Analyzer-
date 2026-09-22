# Developer Guide: Generating Custom Test Signals

This guide provides complete instructions, exact Python API signatures, and executable recipes for synthesizing custom `.iq` and `.wav` RF test files using the **Signal Analyzer** core modules.

---

## Table of Contents
1. [Core Architecture & Signal Pipeline](#1-core-architecture--signal-pipeline)
2. [File Naming & Data Type Conventions](#2-file-naming--data-type-conventions)
3. [Module API Reference & Function Signatures](#3-module-api-reference--function-signatures)
   - [Forward Error Correction (`modulation.fec`, `modulation.rs_fec`, `modulation.ldpc`, `modulation.concatenated`)](#fec-modules)
   - [Interleavers (`modulation.deinterleaver`)](#interleaver-modules)
   - [Sync Words & Correlation (`modulation.correlator`)](#sync-words--correlation)
   - [Modulators & Pulse Shaping (`modulation.signal_gen`)](#modulation--signal-generation)
4. [End-to-End Generator Script (Verified)](#4-end-to-end-generator-script-verified)
5. [How to Customize Schemes & Modulations](#5-how-to-customize-schemes--modulations)
   - [A. Reed-Solomon RS(64, 48)](#a-reed-solomon-rs64-48)
   - [B. LDPC (128, 64)](#b-ldpc-128-64)
   - [C. Concatenated RS + Viterbi](#c-concatenated-rs--viterbi)
   - [D. 16-QAM and 2-FSK Modulations](#d-16-qam-and-2-fsk-modulations)
6. [Embedding Human-Readable ASCII Payloads](#6-embedding-human-readable-ascii-payloads)

---

## 1. Core Architecture & Signal Pipeline

A complete synthetic transmission packet follows this transmit chain:

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Payload Bits   │ ──> │   FEC Encoder    │ ──> │   Interleaver    │
│ (e.g. 512 bits) │     │ (Conv / RS / LDPC│     │ (Block/Diag/Conv)│
└─────────────────┘     └──────────────────┘     └──────────────────┘
                                                           │
┌─────────────────┐     ┌──────────────────┐               │
│ Output .iq/.wav │ <── │  Pulse Shaping   │ <── ┌──────────────────┐
│ (with AWGN/CFO) │     │  & Modulator     │     │ Prepend 32-bit   │
└─────────────────┘     │ (BPSK/QPSK/16QAM)│     │ Sync Word Marker │
                        └──────────────────┘     └──────────────────┘
```

---

## 2. File Naming & Data Type Conventions

[`sig_io/iq_reader.py`](file:///e:/sih147/signal_analyzer/sig_io/iq_reader.py) automatically infers the data type from the filename suffix:

| Suffix / Extension | Target Format | Complex Data Layout | Scaling Factor |
| :--- | :--- | :--- | :--- |
| **`_f32.iq`** or **`.iq`** | `float32` (default) | Interleaved `[I0, Q0, I1, Q1, ...]` 32-bit float | $1.0$ |
| **`_f64.iq`** | `float64` | Interleaved 64-bit float | $1.0$ |
| **`_i16.iq`** / **`_s16.iq`** | `int16` | Interleaved 16-bit signed integer | $1.0 / 32768.0$ |
| **`_i8.iq`** / **`_s8.iq`** | `int8` | Interleaved 8-bit signed integer | $1.0 / 128.0$ |
| **`_u8.iq`** / **`.cu8`** | `uint8` | Interleaved 8-bit unsigned integer (RTL-SDR) | $(x - 128) / 128.0$ |
| **`.wav`** | RIFF WAV | 2-channel stereo (Left = I, Right = Q) | Auto from header |

### Companion Ground-Truth Bits File
To enable **Ground-truth BER calculation** in the GUI, save the original unencoded information bits alongside the `.iq` file as **`<basename>_bits.npy`**:
- `samples/my_test_f32.iq`
- `samples/my_test_f32_bits.npy` *(1D uint8 or int array of information bits)*

---

## 3. Module API Reference & Function Signatures

### FEC Modules

#### 1. Convolutional Code ($K=7, \text{Rate } 1/2$) — `modulation.fec`
```python
def conv_encode(
    bits: np.ndarray | list[int],
    g1: int = 0o171,      # Octal 171 (NASA standard)
    g2: int = 0o133,      # Octal 133 (NASA standard)
    add_tail: bool = True # Adds K-1=6 zero flush bits (output length = 2*(len(bits)+6))
) -> np.ndarray: ...

def viterbi_decode(
    bits: np.ndarray | list[int],
    g1: int = 0o171,
    g2: int = 0o133,
) -> np.ndarray: ...
```

#### 2. Reed-Solomon over $\text{GF}(2^8)$ — `modulation.rs_fec`
```python
def rs_encode(
    data_bytes: np.ndarray | bytes | list[int], # Length must be exactly k bytes
    n: int = 64,                                # Codeword length in bytes (n <= 255)
    k: int = 48,                                # Message length in bytes
    fcr: int = 0                                # First consecutive root exponent
) -> np.ndarray: ...                            # Returns uint8 array of length n

def rs_decode(
    codeword_bytes: np.ndarray | bytes | list[int], # Length n bytes
    n: int = 64,
    k: int = 48,
    fcr: int = 0
) -> np.ndarray: ...                            # Returns uint8 array of length k
```

#### 3. LDPC Code (Rate 1/2) — `modulation.ldpc`
```python
def build_ldpc_code(
    n: int = 128,
    k: int = 64,
    dv: int = 3,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]: ... # Returns (H, G) matrices

def ldpc_encode(
    info_bits: np.ndarray,      # Length k (or multiple of k)
    H: np.ndarray | None = None,
    G: np.ndarray | None = None # Defaults to precomputed (128, 64) code
) -> np.ndarray: ...            # Returns length n bits

def ldpc_decode(
    received_bits: np.ndarray,
    H: np.ndarray | None = None,
    max_iterations: int = 50
) -> tuple[np.ndarray, bool, int]: ... # Returns (decoded_bits, success, iters)
```

#### 4. Concatenated RS + Convolutional — `modulation.concatenated`
```python
def concatenated_encode(
    info_data: np.ndarray | bytes | list[int], # k bytes or k*8 binary bits
    n: int = 64,
    k: int = 48
) -> np.ndarray: ...                           # Returns 2*(n*8 + 6) coded bits

def concatenated_decode(
    received_bits: np.ndarray | list[int],
    n: int = 64,
    k: int = 48,
    return_intermediate: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]: ...
```

---

### Interleaver Modules (`modulation.deinterleaver`)

```python
# 1. Block Interleaver
def block_interleave(bits: np.ndarray | list[int], rows: int = 8, cols: int = 32) -> np.ndarray: ...
def block_deinterleave(bits: np.ndarray | list[int], rows: int = 8, cols: int = 32, original_len: int | None = None) -> np.ndarray: ...

# 2. Convolutional (Forney) Interleaver
def conv_interleave(bits: np.ndarray | list[int], depth: int = 8, span: int = 4) -> np.ndarray: ...
def conv_deinterleave(bits: np.ndarray | list[int], depth: int = 8, span: int = 4, trim_delay: bool = True, original_len: int | None = None) -> np.ndarray: ...

# 3. True Diagonal Interleaver (Deterministic cyclic shift)
def true_diagonal_interleave(bits: np.ndarray | list[int], rows: int = 16, cols: int = 16) -> np.ndarray: ...
def true_diagonal_deinterleave(bits: np.ndarray | list[int], rows: int = 16, cols: int = 16, original_len: int | None = None) -> np.ndarray: ...

# 4. Pseudo-Random Diagonal Interleaver
def diagonal_interleave(bits: np.ndarray | list[int], block_size: int = 256, seed: int = 42) -> np.ndarray: ...
def diagonal_deinterleave(bits: np.ndarray | list[int], block_size: int = 256, seed: int = 42) -> np.ndarray: ...
```

---

### Sync Words & Correlation (`modulation.correlator`)

```python
DEFAULT_SYNC_32 = np.array([...]) # 0xEB902A3C
SYNC_CCSDS_32   = np.array([...]) # 0x1ACFFC1D (Alt A)
SYNC_ALT_B_32   = np.array([...]) # 0xFAF334BE (Alt B)
SYNC_ALT_C_32   = np.array([...]) # 0x352EF853 (Alt C)

def hex_to_sync_bits(hex_val: str, bit_length: int = 32) -> np.ndarray: ...
def parse_sync_word(sync_input: np.ndarray | list[int] | str | None) -> np.ndarray: ...
```

---

### Modulation & Signal Generation (`modulation.signal_gen`)

```python
# Helper functions for baseband signal shaping
def _rrc_filter(beta: float = 0.35, span: int = 8, sps: int = 8) -> np.ndarray: ...
def _apply_offsets(sig: np.ndarray, sample_rate: int, freq_offset_hz: float = 0.0, phase_offset_rad: float = 0.0) -> np.ndarray: ...
def _add_awgn(sig: np.ndarray, snr_db: float) -> np.ndarray: ...

# High-level direct bitstream modulators
def generate_16qam_from_bits(bits: np.ndarray, sps: int = 8, sample_rate: int = 1_000_000, snr_db: float = 15.0, freq_offset_hz: float = 0.0, phase_offset_rad: float = 0.0) -> np.ndarray: ...
def generate_2fsk_from_bits(bits: np.ndarray, sps: int = 8, sample_rate: int = 1_000_000, snr_db: float = 15.0, freq_offset_hz: float = 0.0, phase_offset_rad: float = 0.0, deviation_hz: float = 25000) -> np.ndarray: ...
```

---

## 4. End-to-End Generator Script (Verified)

Save the following script as `generate_custom_packet.py` and run it:

```python
import os
import sys
import numpy as np
from scipy.io import wavfile

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modulation.fec import conv_encode
from modulation.deinterleaver import block_interleave
from modulation.correlator import DEFAULT_SYNC_32
from modulation.signal_gen import _rrc_filter, _apply_offsets, _add_awgn


def create_bpsk_test_signal(
    ascii_message: str = "Hello from Signal Analyzer RF Pipeline!",
    output_basename: str = "custom_demo",
    sample_rate: int = 1_000_000,
    sps: int = 8,
    snr_db: float = 18.0,
    freq_offset_hz: float = 250.0,    # 250 Hz Carrier Frequency Offset
    phase_offset_rad: float = 0.35,   # Phase offset
):
    print(f"1. Encoding Message: {ascii_message!r}")
    raw_bytes = np.frombuffer(ascii_message.encode("utf-8"), dtype=np.uint8)
    info_bits = np.unpackbits(raw_bytes)

    # 2. Convolutional Encoding (K=7, Rate 1/2)
    coded_bits = conv_encode(info_bits, add_tail=True)

    # 3. Pad to multiple of block size (8 x 32 = 256 bits)
    block_size = 8 * 32
    n_pad = (block_size - (len(coded_bits) % block_size)) % block_size
    padded_coded = np.concatenate([coded_bits, np.zeros(n_pad, dtype=int)])

    # 4. Block Interleaving (8 x 32)
    interleaved = block_interleave(padded_coded, rows=8, cols=32)

    # 5. Prepend 32-bit Sync Word (0xEB902A3C)
    sync_word = DEFAULT_SYNC_32
    packet_bits = np.concatenate([sync_word, interleaved])

    # 6. BPSK Modulation + Pulse Shaping
    symbols = 2 * packet_bits - 1  # ±1
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols

    h = _rrc_filter(beta=0.35, span=8, sps=sps)
    sig = np.convolve(upsampled, h, mode="same")
    sig = _apply_offsets(sig, sample_rate, freq_offset_hz, phase_offset_rad)
    sig = _add_awgn(sig, snr_db)

    # 7. Save as float32 .iq file
    iq_path = f"{output_basename}_f32.iq"
    raw_iq = np.empty(2 * len(sig), dtype=np.float32)
    raw_iq[0::2] = sig.real.astype(np.float32)
    raw_iq[1::2] = sig.imag.astype(np.float32)
    raw_iq.tofile(iq_path)

    # 8. Save companion ground-truth bits file for BER validation
    gt_path = f"{output_basename}_f32_bits.npy"
    np.save(gt_path, info_bits)

    # 9. Save as 2-channel Stereo WAV (Left=I, Right=Q)
    wav_path = f"{output_basename}.wav"
    wav_data = np.empty((len(sig), 2), dtype=np.int16)
    wav_data[:, 0] = np.clip(sig.real * 15000, -32767, 32767).astype(np.int16)
    wav_data[:, 1] = np.clip(sig.imag * 15000, -32767, 32767).astype(np.int16)
    wavfile.write(wav_path, sample_rate, wav_data)

    print(f"\n[Generated Successfully]")
    print(f"  - IQ File         : {iq_path} ({os.path.getsize(iq_path):,} bytes)")
    print(f"  - Ground Truth NPY: {gt_path} ({len(info_bits)} bits)")
    print(f"  - Stereo WAV File : {wav_path}")
    return iq_path, gt_path, wav_path


if __name__ == "__main__":
    create_bpsk_test_signal()
```

---

## 5. How to Customize Schemes & Modulations

### A. Reed-Solomon RS(64, 48)
```python
from modulation.rs_fec import rs_encode
from modulation.correlator import DEFAULT_SYNC_32

# Payload must be exactly k=48 bytes
info_bytes = np.random.randint(0, 256, 48, dtype=np.uint8)
cw_bytes = rs_encode(info_bytes, n=64, k=48)
cw_bits = np.unpackbits(cw_bytes)

# Prepend sync word -> modulate
packet_bits = np.concatenate([DEFAULT_SYNC_32, cw_bits])
```

### B. LDPC (128, 64)
```python
from modulation.ldpc import ldpc_encode, DEFAULT_H_128_64, DEFAULT_G_128_64
from modulation.correlator import DEFAULT_SYNC_32

# Payload must be length k=64 bits
info_bits = np.random.randint(0, 2, 64, dtype=np.uint8)
cw_bits = ldpc_encode(info_bits, G=DEFAULT_G_128_64)

packet_bits = np.concatenate([DEFAULT_SYNC_32, cw_bits])
```

### C. Concatenated RS + Viterbi
```python
from modulation.concatenated import concatenated_encode
from modulation.correlator import DEFAULT_SYNC_32

# Payload must be k=48 bytes (or 384 bits)
info_bytes = np.random.randint(0, 256, 48, dtype=np.uint8)
cw_bits = concatenated_encode(info_bytes, n=64, k=48)

packet_bits = np.concatenate([DEFAULT_SYNC_32, cw_bits])
```

### D. 16-QAM and 2-FSK Modulations
```python
from modulation.signal_gen import generate_16qam_from_bits, generate_2fsk_from_bits

# 16-QAM modulation (Gray mapped, unit power normalized)
sig_16qam = generate_16qam_from_bits(packet_bits, sps=8, sample_rate=1_000_000, snr_db=22.0)

# 2-FSK modulation (continuous phase frequency shift keying)
sig_2fsk = generate_2fsk_from_bits(packet_bits, sps=8, sample_rate=1_000_000, snr_db=18.0, deviation_hz=25000)
```

---

## 6. Embedding Human-Readable ASCII Payloads

To recover plain English sentences during live demonstrations:

```python
# Convert ASCII text to bits
message = "Mission Status: Satellite Telemetry Nominal [CAN-7USAT]"
msg_bytes = np.frombuffer(message.encode("utf-8"), dtype=np.uint8)
info_bits = np.unpackbits(msg_bytes)

# Encode -> Transmit -> Receive -> Decode
# To recover text after decoding in Python:
recovered_bytes = np.packbits(decoded_bits[:len(info_bits)]).tobytes()
recovered_text = recovered_bytes.decode("utf-8", errors="replace")
print("Decoded String:", recovered_text)
```
