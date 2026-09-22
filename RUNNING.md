# Running Signal Analyzer

This guide explains how to launch and use the Signal Analyzer GUI and CLI tools.

---

## 1. Prerequisites & Environment Setup

Ensure you have **Python 3.10+** installed.

### Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate on Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Activate on Windows (Command Prompt):
.\venv\Scripts\activate.bat

# Activate on Linux / macOS:
source venv/bin/activate
```

### Install Dependencies

```bash
pip install PySide6 numpy scipy matplotlib scikit-learn
```

---

## 2. Bundled Fonts & Styling

* **Typography**: The UI uses **Inter** for user interface elements and **IBM Plex Mono** for numerical readouts, bitstreams, and metrics.
* **Auto-Loading**: The required font files (`Inter-Variable.ttf`, `IBMPlexMono-Regular.ttf`, `IBMPlexMono-SemiBold.ttf`) are bundled in `gui/fonts/` and registered into the Qt application font database (`QFontDatabase`) automatically on startup. No manual font installation is necessary.
* **Theme**: The styling is centrally defined in [`gui/style.qss`](file:///e:/sih147/signal_analyzer/gui/style.qss).

---

## 3. Launching the Application

### A. Graphical User Interface (GUI)

```bash
# Launch empty analyzer GUI:
python main.py

# Launch with a demo signal pre-loaded:
python main.py samples/true_diagonal_bpsk_demo.iq
```

### B. Headless CLI Mode (Generate & Save Diagnostic Plots)

```bash
# Save Power Spectral Density, Waterfall/Spectrogram, and Constellation PNGs for all sample signals:
python main.py --headless all

# Generate plots for a specific modulation:
python main.py --headless bpsk
python main.py --headless qpsk
python main.py --headless 2fsk
```
Generated plot images are saved directly to the [`samples/`](file:///e:/sih147/signal_analyzer/samples) directory.
