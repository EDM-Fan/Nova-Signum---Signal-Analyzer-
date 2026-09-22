"""
main.py  —  Signal Analyser launcher

Usage:
    python main.py                        # open GUI (no file)
    python main.py samples/bpsk_*.iq     # open GUI with file pre-loaded
    python main.py --headless SIGNAL     # save PNG plots without GUI
                                          # SIGNAL = bpsk | qpsk | 2fsk | all
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Headless mode: generate and save plots for all 3 signals
# ---------------------------------------------------------------------------

def run_headless(signal: str = "all"):
    """Save Spectrum + Waterfall + Constellation PNGs without the GUI."""
    import numpy as np
    from sig_io.iq_reader    import read_iq
    from dsp.spectrum        import analyse, plot_spectrum
    from dsp.constellation   import plot_constellation
    from dsp.waterfall       import plot_waterfall

    samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    FS = 1_000_000

    targets = {
        "bpsk": "bpsk_1000ksps_f32.iq",
        "qpsk": "qpsk_1000ksps_f32.iq",
        "2fsk": "2fsk_1000ksps_f32.iq",
    }

    to_run = targets if signal == "all" else {signal: targets[signal]}

    for name, fname in to_run.items():
        path = os.path.join(samples_dir, fname)
        if not os.path.isfile(path):
            print(f"[SKIP] {fname} not found")
            continue

        print(f"\n=== {name.upper()} ===")
        info = read_iq(path, sample_rate=FS)
        samples = info["samples"]
        label   = name.upper()

        # Spectrum
        result = analyse(samples, FS, nfft=8192)
        plot_spectrum(result,
                      title=f"{label} — Power Spectral Density",
                      save_path=os.path.join(samples_dir, f"{name}_spectrum.png"))

        # Waterfall
        plot_waterfall(samples, FS,
                       title=f"{label} — Waterfall / Spectrogram",
                       save_path=os.path.join(samples_dir, f"{name}_waterfall.png"),
                       nfft=1024, hop=256)

        # Constellation
        plot_constellation(samples, FS,
                           title=f"{label} — Constellation (I/Q)",
                           save_path=os.path.join(samples_dir, f"{name}_constellation.png"))

        print(f"  SNR ~{result['snr_db']:.1f} dB | "
              f"OccBW {result['occupied_bw']/1e3:.2f} kHz | "
              f"Centre {result['center_freq']:+.1f} Hz")

    print("\n[OK] All plots saved to samples/")


# ---------------------------------------------------------------------------
# GUI mode
# ---------------------------------------------------------------------------

def run_gui(preload_path: str | None = None):
    import time
    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] [GUI] {msg}", flush=True)

    log("Starting PySide6 application...")
    try:
        from PySide6.QtWidgets import QApplication
        log("Imported PySide6.QtWidgets")

        # Set up an uncaught exception hook so errors aren't swallowed
        def handle_exception(exc_type, exc_value, exc_traceback):
            import traceback
            log("UNCAUGHT EXCEPTION:")
            traceback.print_exception(exc_type, exc_value, exc_traceback)

        sys.excepthook = handle_exception

        log("Creating QApplication instance...")
        app = QApplication.instance() or QApplication(sys.argv)
        log(f"QApplication created (platform: {app.platformName()})")

        log("Importing MainWindow from gui.main_window...")
        from gui.main_window import MainWindow

        log("Instantiating MainWindow...")
        win = MainWindow()
        win.setGeometry(80, 80, 1280, 840)
        log("MainWindow instantiated successfully.")

        log("Showing MainWindow window...")
        win.showNormal()
        win.show()
        win.raise_()
        win.activateWindow()
        app.processEvents()

        geom = win.geometry()
        frame = win.frameGeometry()
        log(f"Window Geometry: x={geom.x()}, y={geom.y()}, w={geom.width()}, h={geom.height()}")
        log(f"Frame Geometry : x={frame.x()}, y={frame.y()}, w={frame.width()}, h={frame.height()}")
        log(f"isVisible: {win.isVisible()}, isHidden: {win.isHidden()}, windowState: {win.windowState()}")
        
        screen = win.screen()
        if screen:
            s_geom = screen.geometry()
            log(f"Screen name: '{screen.name()}', Screen Geometry: x={s_geom.x()}, y={s_geom.y()}, w={s_geom.width()}, h={s_geom.height()}")
        else:
            log("Screen: None detected")

        if preload_path:
            abs_preload = os.path.abspath(preload_path)
            if os.path.isfile(abs_preload):
                log(f"Pre-loading file: {abs_preload}")
                win.load_file(abs_preload)
            else:
                log(f"[WARN] Preload file not found: {preload_path}")

        log("Entering Qt event loop (app.exec)... Window should now be visible.")
        exit_code = app.exec()
        log(f"Qt event loop exited with code: {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        import traceback
        log(f"FAILED to launch GUI: {e}")
        traceback.print_exc()
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--headless" in args:
        args.remove("--headless")
        signal = args[0] if args else "all"
        run_headless(signal)
    else:
        preload = args[0] if args else None
        run_gui(preload)

