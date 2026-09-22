import sys, os, time
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

app = QApplication.instance() or QApplication(sys.argv)
win = MainWindow()

def wait_for_load():
    t0 = time.time()
    while not win.btn_classify.isEnabled():
        app.processEvents()
        time.sleep(0.05)
        if time.time() - t0 > 15.0:
            raise TimeoutError("Timed out waiting for file load.")

def wait_pipeline(timeout=15.0):
    t0 = time.time()
    while win._pipeline_thread is not None and win._pipeline_thread.isRunning():
        app.processEvents()
        time.sleep(0.01)
        if time.time() - t0 > timeout:
            raise TimeoutError("Timed out waiting for pipeline worker.")
    app.processEvents()
    time.sleep(0.05)

print("=" * 60)
print("TEST 1: Alternate Sync Word (Alt A: 0x1ACFFC1D)")
print("=" * 60)
alt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "bpsk_sync_alt_demo.iq")
win.load_file(alt_file)
wait_for_load()

# Classify
win._classify()
wait_pipeline()
print("Classify:", win.lbl_class.text())

# Demodulate
win._demodulate()
wait_pipeline()
print("Demod:", win.lbl_demod.toPlainText()[:60].replace("\n", " "), "...")

# 1A. Select Alt A from dropdown
idx = win.cmb_sync.findData("0x1ACFFC1D")
win.cmb_sync.setCurrentIndex(idx)
print(f"Selected sync dropdown: '{win.cmb_sync.currentText()}' (data={win.cmb_sync.currentData()})")

# Find Sync
win._find_sync()
wait_pipeline()
print("\n--- EXACT SYNC LABEL (Alt A selected) ---")
print(win.lbl_sync.toPlainText())

# 1B. Test custom typed hex value
win.cmb_sync.setEditText("0x1ACFFC1D")
print(f"\nTyped custom sync text: '{win.cmb_sync.currentText()}'")
win._find_sync()
wait_pipeline()
print("\n--- EXACT SYNC LABEL (Custom hex typed) ---")
print(win.lbl_sync.toPlainText())

print("\n" + "=" * 60)
print("TEST 2: Existing Default Sync File (bpsk_sync_demo.iq)")
print("=" * 60)
def_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "bpsk_sync_demo.iq")
win.load_file(def_file)
wait_for_load()

# Classify
win._classify()
wait_pipeline()
print("Classify:", win.lbl_class.text())

# Demodulate
win._demodulate()
wait_pipeline()
print("Demod:", win.lbl_demod.toPlainText()[:60].replace("\n", " "), "...")

# Select Default in dropdown
idx_def = win.cmb_sync.findData("0xEB902A3C")
win.cmb_sync.setCurrentIndex(idx_def)
print(f"Selected sync dropdown: '{win.cmb_sync.currentText()}' (data={win.cmb_sync.currentData()})")

# Find Sync
win._find_sync()
wait_pipeline()
print("\n--- EXACT SYNC LABEL (Default sync) ---")
print(win.lbl_sync.toPlainText())
