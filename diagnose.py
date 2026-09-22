"""
diagnose.py — Run this FIRST to find out exactly where startup hangs.
Each stage prints a timestamped line. The last line printed = where it died.

Run with:
    python diagnose.py
"""
import sys, os, time

def ts(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

ts("START — Python OK")
ts(f"Python: {sys.executable}")
ts(f"CWD   : {os.getcwd()}")

# ── 1. Core stdlib ──────────────────────────────────────────────────────────
ts("Importing numpy ...")
import numpy as np
ts(f"  numpy {np.__version__} OK")

ts("Importing scipy ...")
import scipy
ts(f"  scipy {scipy.__version__} OK")

ts("Importing matplotlib ...")
import matplotlib
ts(f"  matplotlib {matplotlib.__version__} OK")

# ── 2. Project modules ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

ts("Importing sig_io.iq_reader ...")
from sig_io.iq_reader import read_iq
ts("  OK")

ts("Importing dsp.spectrum ...")
from dsp.spectrum import analyse
ts("  OK")

ts("Importing dsp.constellation ...")
from dsp.constellation import prepare_samples
ts("  OK")

ts("Importing dsp.waterfall ...")
from dsp.waterfall import compute_stft
ts("  OK")

# ── 3. Matplotlib backend ────────────────────────────────────────────────────
ts("Setting matplotlib backend to QtAgg ...")
matplotlib.use("QtAgg")
ts("  Backend set OK")

ts("Importing matplotlib.pyplot ...")
import matplotlib.pyplot as plt
ts("  OK")

# ── 4. PySide6 core ──────────────────────────────────────────────────────────
ts("Importing PySide6.QtCore ...")
from PySide6.QtCore import Qt, __version__ as qt_ver
ts(f"  PySide6/Qt {qt_ver} OK")

ts("Importing PySide6.QtWidgets ...")
from PySide6.QtWidgets import QApplication
ts("  OK")

# ── 5. QApplication ──────────────────────────────────────────────────────────
ts("Creating QApplication ...")
app = QApplication.instance() or QApplication(sys.argv)
ts("  QApplication created OK")

ts("Checking platform plugin ...")
ts(f"  Platform: {app.platformName()}")

# ── 6. Matplotlib Qt canvas ──────────────────────────────────────────────────
ts("Importing FigureCanvasQTAgg ...")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
ts("  FigureCanvasQTAgg OK")

# ── 7. MainWindow ────────────────────────────────────────────────────────────
ts("Importing gui.main_window ...")
from gui.main_window import MainWindow
ts("  Import OK")

ts("Creating MainWindow ...")
win = MainWindow()
ts("  MainWindow created OK")

ts("Calling win.show() ...")
win.show()
ts("  show() returned — entering event loop for 2 seconds ...")

from PySide6.QtCore import QTimer
QTimer.singleShot(2000, app.quit)
app.exec()
ts("  Event loop exited cleanly")

ts("ALL STAGES PASSED — run: python main.py")
