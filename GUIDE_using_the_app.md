# User Guide: Signal Analyzer GUI

Welcome to the **Signal Analyzer** — an end-to-end RF signal inspection, modulation classification, synchronization, and Forward Error Correction (FEC) decoding suite built with PySide6, NumPy, SciPy, and Matplotlib.

This guide walks you through using the application from first launch to exporting decoded telemetry and reports.

---

## Table of Contents
1. [Launching the Application](#1-launching-the-application)
2. [Interface Overview](#2-interface-overview)
3. [Step 1: Loading a Signal File](#3-step-1-loading-a-signal-file)
4. [Step 2: Classify Modulation](#4-step-2-classify-modulation)
5. [Step 3: Demodulate Bitstream](#5-step-3-demodulate-bitstream)
6. [Step 4: Frame Synchronization (Find Sync)](#6-step-4-frame-synchronization-find-sync)
7. [Step 5: FEC Auto-Detection & Manual Override](#7-step-5-fec-auto-detection--manual-override)
8. [Step 6: FEC Decoding & Metric Verification](#8-step-6-fec-decoding--metric-verification)
9. [Step 7: Reading the Bitstream Viewport](#9-step-7-reading-the-bitstream-viewport)
10. [Step 8: Exporting Data & Generating Reports](#10-step-8-exporting-data--generating-reports)
11. [Step 9: Resetting State & Keyboard Shortcuts](#11-step-9-resetting-state--keyboard-shortcuts)

---

## 1. Launching the Application

Ensure your Python virtual environment is active, then run:

```bash
python main.py
```

*(Alternatively: `python gui/main_window.py`)*

Upon launch, you are greeted by the **Welcome Screen** with a quick-start checklist, supported format cards, and drag-and-drop zone.

---

## 2. Interface Overview

The interface is structured in a **3-column RF console layout**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ [●] SIGNAL ANALYZER    Loaded — bpsk_demo.iq | 1.000 MHz fs | 10,496 samples    [● Ready]│
├─────────────────┬─────────────────────────────────────────────────┬────────────────────┤
│ PIPELINE        │ CENTER VIEWPORT (Tabs: 1..4)                   │ METRICS & ACTIONS  │
│ [1] File Load   │  [Spectrum] [Waterfall] [Constellation] [Bitstream] │ Decoded Bits: 640  │
│ [2] Classify    │ ┌─────────────────────────────────────────────┐ │ Code Rate:    1/2  │
│ [3] Demodulate  │ │                                             │ │ GT BER:       0.0% │
│ [4] Find Sync   │ │          Interactive RF Visualizer          │ │ Parity Match: 97.3%│
│ [5] FEC Detect  │ │                                             │ ├────────────────────┤
│ [6] Decode      │ └─────────────────────────────────────────────┘ │ Auto-Detect FEC    │
│                 │ FOOTER READOUT: SNR | CFO | Phase | OccBW | Samps│ [Export] [Report]  │
└─────────────────┴─────────────────────────────────────────────────┴────────────────────┘
```

- **Left Column (Pipeline Stepper)**: Guides you step-by-step through the 6 processing stages.
- **Center Viewport**: Displays high-resolution plots (Spectrum, Waterfall, Constellation) and the complete Bitstream viewer.
- **Right Column**: Displays quantitative decode metrics, FEC detection breakdown, and action controls.

---

## 3. Step 1: Loading a Signal File

### Supported Formats
- **Raw IQ (`.iq`, `.bin`)**: Float32 (`_f32.iq`), Float64 (`_f64.iq`), Signed 16-bit PCM (`_i16.iq`, `_s16.iq`), Signed 8-bit (`_i8.iq`), Unsigned 8-bit (`_u8.iq`, `.cu8`).
- **Baseband Audio / WAV (`.wav`)**: 2-channel stereo (I on Left, Q on Right) or 1-channel real.

### How to Load
1. Click **Open File** (`Ctrl+O`) in the left panel, or drag and drop any signal file directly onto the window.
2. Select your sample rate from the **Sample Rate** dropdown (defaults to `Fs = 1 MHz` for raw `.iq` files; WAV files read their sample rate automatically from the file header).
3. **Suggested Rate (Auto-Estimation)**: The analyzer automatically computes a cyclostationary baud rate estimation on load. If a different sample rate is recommended, an **Apply Fs** button lights up with the suggested frequency. Click **Apply Fs** to reload with the suggested rate.

Once loaded, the **Spectrum**, **Waterfall**, and **Constellation** tabs immediately render, and the top-right status pill displays `● File Loaded`.

---

## 4. Step 2: Classify Modulation

Click **Classify Modulation** in the left sidebar.

- The machine learning classifier evaluates higher-order cumulants ($C_{40}, C_{42}$), spectral symmetry, envelope variation, and instantaneous frequency kurtosis.
- **Recognized Schemes**: `BPSK`, `QPSK`, `16QAM`, `2FSK`.
- **Confidence Tiers & Alerts**:
  - $\ge 70\%$ confidence: Normal green pass.
  - $50\% \le \text{confidence} < 70\%$: Amber warning banner (`⚠ Low classification confidence`).
  - $< 50\%$ confidence: Red error banner (`⚠ Very low classification confidence — result unreliable`).

---

## 5. Step 3: Demodulate Bitstream

Click **Demodulate**.

- Applies optimal carrier frequency compensation, symbol clock alignment, and Gray-coded constellation demapping according to the classified modulation type.
- The raw recovered bitstream is populated into the **Bitstream** tab and stage readout.

---

## 6. Step 4: Frame Synchronization (Find Sync)

Raw demodulated bitstreams contain arbitrary bit offsets and phase ambiguities (such as $180^\circ$ phase inversion in BPSK, or $90^\circ/180^\circ/270^\circ$ quadrant rotations in QPSK).

### The Sync Word Selector (`cmb_sync`)
Before clicking **Find Sync**, ensure the **Sync Word** dropdown matches the packet protocol used in your capture:
- **Default (`0xEB902A3C`)**: Standard 32-bit frame sync marker.
- **Alt A CCSDS (`0x1ACFFC1D`)**: Standard space packet sync word (CCSDS 131.0-B-3).
- **Alt B (`0xFAF334BE`)**: High-contrast preamble marker.
- **Alt C (`0x352EF853`)**: Alternate Barker/Gold code sequence.
- **Custom Hex**: You can type any 32-bit hex value directly into the combo box.

### Dual-Check Confidence Gate
When you click **Find Sync**, the engine runs a dual-verification check:
1. **Correlation Score Threshold**: Correlation match must be $\ge 75\%$ to avoid false triggers on random payload bits.
2. **Time-Domain Signal Energy Check**: For bursty files, the candidate sync location is checked against the baseline time-domain noise floor. If the correlator locked onto a quiet noise region outside the true transmission burst, it triggers:
   - Status Pill: `⚡ Suspect Sync (Noise Region @ bit N)`
   - Error Banner: `Sync lock is in a noise-floor region (local signal power at or below noise floor). Lock is spurious.`

When sync is verified, carrier phase ambiguity is resolved (e.g., `direct (0 deg)` or `inverted (180 deg)`), and the synchronized payload bits are forwarded to the FEC stage.

---

## 7. Step 5: FEC Auto-Detection & Manual Override

Click **Auto-Detect FEC**.

The blind FEC analyzer tests the synchronized payload against multiple code families:
1. **Reed-Solomon (RS)**: Algebraic syndrome verification over $\text{GF}(2^8)$ for sizes like $\text{RS}(64,48)$ and $\text{RS}(32,24)$.
2. **LDPC**: Normalized Min-Sum Belief Propagation syndrome convergence ($H \cdot c^T = 0 \pmod 2$) on Rate-1/2 $(128, 64)$ parity graphs.
3. **Concatenated (RS + Viterbi)**: Inner convolutional trellis decoding followed by outer Reed-Solomon syndrome clearance.
4. **Convolutional ($K=7, \text{Rate } 1/2$)**: Evaluated across 4 de-interleaver architectures:
   - **Block Interleaver ($8 \times 32$)**
   - **Convolutional / Forney ($8 \times 4$)**
   - **True Diagonal ($16 \times 16$)**
   - **Pseudo-Random Diagonal**

- The top match and runner-up with confidence percentages are displayed in the **Auto-Detect FEC Summary Card**.
- **Manual Override**: You can change the FEC scheme or Reed-Solomon $(n, k)$ codeword size at any time using the **FEC Scheme** and **RS Size** dropdowns.

---

## 8. Step 6: FEC Decoding & Metric Verification

Click **Decode** (`Ctrl+D`).

The decoder corrects channel bit errors and updates the **Decode Metrics**:

| Metric | Description |
| :--- | :--- |
| **Decoded Bits** | Number of recovered information bits (e.g. `640` bits). |
| **Code Rate** | Transmission code rate (e.g. `1/2` for Conv/LDPC, `48/64` for RS). |
| **Ground-truth BER** | **Only shown for synthetic test files with a companion `<name>_bits.npy` file.** Displays the true Bit Error Rate (e.g. `0.00%`). Real unknown captures show `N/A (real capture)`. |
| **Re-encode Match** | Re-encodes the decoded bits through the forward encoder and compares against the received channel bitstream. Reports the true roundtrip agreement percentage (e.g. `97.3%`). |

### Status Pill Severity
- **`● Pipeline complete` (Green)**: All stages passed with high confidence.
- **`⚠ Pipeline complete (low confidence)` (Amber)**: One or more stages had low confidence scores.
- **`⚡ Pipeline complete (spurious noise lock)` (Red)**: Sync locked onto a noise region.

---

## 9. Step 7: Reading the Bitstream Viewport

Press **`4`** or switch to the **Bitstream** tab in the center viewport. The view is structured hierarchically from top to bottom:

```text
[DECODED PAYLOAD: Rate 1/2 Convolutional (Block Interleaver 8x32) | 640 bits]
0100100001100101011011000110110001101111001011000010000001010011...

[SYNC LOCKED: @ bit 0 (100.0% match, direct (0 deg)) using 0xEB902A3C]
Payload: 1,280 bits
110010100101101011001100...

[RAW DEMODULATED BITSTREAM: 1,312 bits]
111010111001000000101010001111001100101001011010...
```

1. **Top Section (`[DECODED PAYLOAD]`)**: Error-corrected information bits. If an ASCII message was transmitted, packing these bits recovers the plaintext string.
2. **Middle Section (`[SYNC LOCKED]` / `[SUSPECT SYNC]`)**: Raw channel bits following the sync word marker.
3. **Bottom Section (`[RAW DEMODULATED BITSTREAM]`)**: Complete unaligned bitstream straight from the demodulator.

---

## 10. Step 8: Exporting Data & Generating Reports

### Export Data (`_export_results`)
Click **Export Data** in the right sidebar:
- **JSON Export (`.json`)**: Exports complete session metadata, timestamps, RF metrics (SNR, CFO, OccBW), predicted modulation, sync details, and decoded bitstream text.
- **CSV Export (`.csv`)**: Tabular export formatted for spreadsheet review.

### Save Report (`_save_report`)
Click **Save Report** in the right sidebar:
- **PDF Report (`.pdf`)**: Generates a publication-quality 1-page PDF document containing a metrics summary table, high-resolution Spectrum plot, and Waterfall plot.
- **PNG Screenshot (`.png`)**: Captures a full-resolution PNG image of the application console.

---

## 11. Step 9: Resetting State & Keyboard Shortcuts

Click **Reset** (`Ctrl+R`) at the bottom of the left sidebar to clear all loaded data, reset plots, and return the interface to a clean initial state without needing to restart the application.

### Keyboard Shortcuts Reference

| Shortcut | Action |
| :--- | :--- |
| **`Ctrl + O`** | Open signal file dialog |
| **`Ctrl + D`** | Execute FEC Decode |
| **`Ctrl + R`** | Reset application state |
| **`1`** | Switch to **Spectrum** tab |
| **`2`** | Switch to **Waterfall** tab |
| **`3`** | Switch to **Constellation** tab |
| **`4`** | Switch to **Bitstream** tab |
| **`Esc`** | Dismiss active banner alerts |
