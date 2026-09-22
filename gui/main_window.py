"""
gui/main_window.py
==================
PySide6 main window for the Signal Analyzer.

Three-Column Layout:
  1. Left: Vertical Pipeline Stepper (File Load -> Classify -> Demod -> Sync -> FEC/Interleave -> Decode)
  2. Center: Tabbed Viewport (Spectrum | Waterfall | Constellation | Bitstream) with footer readout bar
  3. Right: Metrics, Auto-Detect FEC summary card, and Action Controls
"""

from __future__ import annotations
import os
import sys
import glob
import json
import csv
import datetime
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget,
    QStatusBar, QSizePolicy, QFrame, QComboBox, QTextEdit,
    QProgressBar, QScrollArea, QGraphicsOpacityEffect,
    QStackedWidget, QApplication,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, Property, QEasingCurve, QPropertyAnimation, QTimer
from PySide6.QtGui import QFont, QFontDatabase, QColor, QPainter, QBrush, QPen, QShortcut, QKeySequence, QDragEnterEvent, QDropEvent

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib

# Project root on path (one level up from gui/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def register_fonts():
    """Register bundled fonts (Inter, IBM Plex Mono) into QFontDatabase."""
    from PySide6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None:
        return
    fonts_dir = os.path.join(_ROOT, "gui", "fonts")
    if os.path.isdir(fonts_dir):
        for font_file in glob.glob(os.path.join(fonts_dir, "*.ttf")):
            QFontDatabase.addApplicationFont(font_file)


# Initialize Matplotlib rcParams for sleek near-black RF console theme
matplotlib.rcParams.update({
    "axes.facecolor":   "#0A0E14",
    "figure.facecolor": "#0A0E14",
    "text.color":       "#E4E9F0",
    "axes.labelcolor":  "#6B7684",
    "xtick.color":      "#6B7684",
    "ytick.color":      "#6B7684",
    "axes.edgecolor":   "#1E2733",
    "grid.color":       "#1E2733",
    "grid.linestyle":   ":",
    "grid.linewidth":   0.6,
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Inter", "Segoe UI", "DejaVu Sans", "Arial"],
})


from sig_io.iq_reader  import read_iq
from sig_io.wav_reader import read_wav
from dsp.spectrum      import analyse      as spec_analyse
from dsp.constellation import prepare_samples, compute_stats
from dsp.waterfall     import compute_stft


# ---------------------------------------------------------------------------
# Matplotlib canvas widget helper
# ---------------------------------------------------------------------------

class MplCanvas(FigureCanvas):
    def __init__(self, width=10, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi,
                          facecolor="#0A0E14", constrained_layout=True)
        super().__init__(self.fig)
        self.setStyleSheet("background-color: #0A0E14;")
        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def clear(self):
        self.fig.clear()
        self.draw()


# ---------------------------------------------------------------------------
# Worker thread so the GUI stays responsive while computing
# ---------------------------------------------------------------------------

class AnalysisWorker(QObject):
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, file_path: str, sample_rate: int = 1_000_000):
        super().__init__()
        self.file_path   = file_path
        self.sample_rate = sample_rate

    def run(self):
        try:
            ext = os.path.splitext(self.file_path)[1].lower()
            if ext == ".wav":
                info = read_wav(self.file_path)
            else:
                info = read_iq(self.file_path, sample_rate=self.sample_rate)

            samples = info["samples"]
            fs      = info["sample_rate"] or self.sample_rate

            spec    = spec_analyse(samples, fs, nfft=8192, window="hann")
            constel = prepare_samples(samples, fs)
            freqs_w, times_w, Sdb_w = compute_stft(samples, fs,
                                                    nfft=1024, hop=256)

            rate_est = None
            if ext != ".wav":
                try:
                    from dsp.rate_estimator import estimate_sample_rate
                    rate_est = estimate_sample_rate(samples)
                except Exception:
                    rate_est = None

            self.finished.emit({
                "info":     info,
                "spec":     spec,
                "constel":  constel,
                "wfall":   (freqs_w, times_w, Sdb_w),
                "fs":       fs,
                "rate_est": rate_est,
            })
        except Exception as exc:
            self.error.emit(str(exc))


class PipelineTaskThread(QThread):
    finished_result = Signal(object)
    error_occurred  = Signal(str)

    def __init__(self, task_fn, parent=None):
        super().__init__(parent)
        self.task_fn = task_fn

    def run(self):
        try:
            print("[PipelineTaskThread] executing task_fn...")
            res = self.task_fn()
            print("[PipelineTaskThread] task_fn returned, emitting signal...")
            self.finished_result.emit(res)
            print("[PipelineTaskThread] signal emitted.")
        except Exception as exc:
            print("[PipelineTaskThread] exception:", exc)
            self.error_occurred.emit(str(exc))


# ---------------------------------------------------------------------------
# Animated Stage Dot Widget
# ---------------------------------------------------------------------------

class AnimatedStageDot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(10, 10)
        self._fill_color = QColor("#0A0E14")
        self._border_color = QColor("#414C59")
        self._anim_fill = None
        self._anim_border = None

    def get_fill_color(self) -> QColor:
        return self._fill_color

    def set_fill_color(self, c: QColor):
        self._fill_color = QColor(c)
        self.update()

    fill_color = Property(QColor, get_fill_color, set_fill_color)

    def get_border_color(self) -> QColor:
        return self._border_color

    def set_border_color(self, c: QColor):
        self._border_color = QColor(c)
        self.update()

    border_color = Property(QColor, get_border_color, set_border_color)

    def transition_to(self, state: str):
        if state == "done":
            target_fill = QColor("#4EE1B8")
            target_border = QColor("#4EE1B8")
        elif state == "active":
            target_fill = QColor("#0A0E14")
            target_border = QColor("#4EE1B8")
        else:  # pending
            target_fill = QColor("#0A0E14")
            target_border = QColor("#414C59")

        self._anim_fill = QPropertyAnimation(self, b"fill_color")
        self._anim_fill.setDuration(350)
        self._anim_fill.setStartValue(self._fill_color)
        self._anim_fill.setEndValue(target_fill)
        self._anim_fill.setEasingCurve(QEasingCurve.OutCubic)

        self._anim_border = QPropertyAnimation(self, b"border_color")
        self._anim_border.setDuration(350)
        self._anim_border.setStartValue(self._border_color)
        self._anim_border.setEndValue(target_border)
        self._anim_border.setEasingCurve(QEasingCurve.OutCubic)

        self._anim_fill.start()
        self._anim_border.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(self._fill_color))
        painter.setPen(QPen(self._border_color, 2))
        painter.drawEllipse(1, 1, 8, 8)


# ---------------------------------------------------------------------------
# Pipeline Stage Widget (Step in the vertical stepper)
# ---------------------------------------------------------------------------

class PipelineStageWidget(QFrame):
    def __init__(self, title: str, initial_sub: str = "Pending"):
        super().__init__()
        self.setProperty("class", "stage-frame")
        self.setProperty("stageState", "pending")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 8, 8, 8)
        lay.setSpacing(12)

        # Dot
        self.dot = AnimatedStageDot(self)
        lay.addWidget(self.dot, alignment=Qt.AlignTop)

        # Text column
        tlay = QVBoxLayout()
        tlay.setContentsMargins(0, 0, 0, 0)
        tlay.setSpacing(2)

        self.lbl_title = QLabel(title)
        self.lbl_title.setProperty("class", "stage-label")
        self.lbl_title.setProperty("stageState", "pending")

        self.lbl_sub = QLabel(initial_sub)
        self.lbl_sub.setProperty("class", "stage-sub")
        self.lbl_sub.setProperty("stageState", "pending")

        tlay.addWidget(self.lbl_title)
        tlay.addWidget(self.lbl_sub)
        lay.addLayout(tlay, stretch=1)

    def set_state(self, state: str, subtext: str | None = None):
        """state in ('pending', 'active', 'done')"""
        self.setProperty("stageState", state)
        self.lbl_title.setProperty("stageState", state)
        self.lbl_sub.setProperty("stageState", state)
        if subtext is not None:
            self.lbl_sub.setText(subtext)
        self.dot.transition_to(state)
        self.style().unpolish(self)
        self.style().polish(self)
        self.lbl_title.style().unpolish(self.lbl_title)
        self.lbl_title.style().polish(self.lbl_title)
        self.lbl_sub.style().unpolish(self.lbl_sub)
        self.lbl_sub.style().polish(self.lbl_sub)


# ---------------------------------------------------------------------------
# Alert Banner Component (Dismissible Warning / Error / Success)
# ---------------------------------------------------------------------------

class AlertBanner(QFrame):
    dismissed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("alertBanner")
        self.setProperty("level", "error")
        self.hide()

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 8, 6)
        lay.setSpacing(10)

        self.lbl_text = QLabel()
        self.lbl_text.setObjectName("bannerText")
        self.lbl_text.setWordWrap(True)
        lay.addWidget(self.lbl_text, stretch=1)

        self.btn_close = QPushButton("✕")
        self.btn_close.setObjectName("btnBannerClose")
        self.btn_close.setFixedSize(24, 24)
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setToolTip("Dismiss alert [Esc]")
        self.btn_close.clicked.connect(self.dismiss)
        lay.addWidget(self.btn_close, alignment=Qt.AlignRight | Qt.AlignVCenter)

    def show_message(self, text: str, level: str = "error"):
        """Display banner with level in ('error', 'warn', 'success')."""
        prefix = "⚠  " if level == "warn" else ("✓  " if level == "success" else "✖  ")
        self.lbl_text.setText(f"{prefix}{text}")
        self.setProperty("level", level)
        self.style().unpolish(self)
        self.style().polish(self)
        self.lbl_text.style().unpolish(self.lbl_text)
        self.lbl_text.style().polish(self.lbl_text)
        self.show()

    def dismiss(self):
        self.hide()
        self.dismissed.emit()


# ---------------------------------------------------------------------------
# Empty State Viewport Placeholder Widget
# ---------------------------------------------------------------------------

class EmptyStateWidget(QFrame):
    open_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("emptyStateWidget")

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(12)

        self.glyph = QLabel("◈")
        self.glyph.setObjectName("emptyStateGlyph")
        self.glyph.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.glyph)

        self.title = QLabel("NO SIGNAL LOADED")
        self.title.setObjectName("emptyStateTitle")
        self.title.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.title)

        self.subtitle = QLabel("Drop an .iq, .wav, or .bin file anywhere, or click Open File")
        self.subtitle.setObjectName("emptyStateSubtitle")
        self.subtitle.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.subtitle)

        lay.addSpacing(6)

        btn_open = QPushButton("Open File (Ctrl+O)")
        btn_open.setObjectName("btnPrimary")
        btn_open.setFixedWidth(160)
        btn_open.setCursor(Qt.PointingHandCursor)
        btn_open.clicked.connect(self.open_requested.emit)
        lay.addWidget(btn_open, alignment=Qt.AlignCenter)

        lay.addSpacing(8)

        self.hints = QLabel("Shortcuts: Ctrl+O Open · Ctrl+D Decode · Ctrl+R Reset · 1–4 Tabs · Esc Close Alert")
        self.hints.setObjectName("emptyStateHint")
        self.hints.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.hints)


# ---------------------------------------------------------------------------
# Drag-and-Drop Highlight Overlay Widget
# ---------------------------------------------------------------------------

class DropOverlay(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dropOverlay")
        self.hide()

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(8)

        lbl = QLabel("⬇  DROP SIGNAL FILE TO LOAD (.iq / .wav / .bin)")
        lbl.setObjectName("dropOverlayText")
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)


# ---------------------------------------------------------------------------
# Splash / Hero Moment Overlay on Launch
# ---------------------------------------------------------------------------

class SplashHeroOverlay(QFrame):
    dismissed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("splashHeroOverlay")
        self.setStyleSheet("""
            QFrame#splashHeroOverlay {
                background-color: #0A0E14;
            }
        """)
        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setCursor(Qt.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(14)

        # Large glowing dot / logo mark
        dot_box = QHBoxLayout()
        dot_box.setAlignment(Qt.AlignCenter)
        self.dot = QLabel()
        self.dot.setFixedSize(14, 14)
        self.dot.setStyleSheet("""
            background-color: #4EE1B8;
            border-radius: 7px;
        """)
        dot_box.addWidget(self.dot)
        lay.addLayout(dot_box)

        # Brand Title
        self.lbl_title = QLabel("SIGNAL ANALYZER")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setFont(QFont("Inter", 24, QFont.Bold))
        self.lbl_title.setStyleSheet("color: #E4E9F0; letter-spacing: 3px;")
        lay.addWidget(self.lbl_title)

        # Subtitle
        self.lbl_sub = QLabel("ADVANCED RF SIGNAL INSPECTION & BLIND FEC DEMODULATION")
        self.lbl_sub.setAlignment(Qt.AlignCenter)
        self.lbl_sub.setFont(QFont("IBM Plex Mono", 10, QFont.DemiBold))
        self.lbl_sub.setStyleSheet("color: #4EE1B8; letter-spacing: 1px;")
        lay.addWidget(self.lbl_sub)

        lay.addSpacing(16)

        # Dismiss hint
        self.lbl_hint = QLabel("Click anywhere or press any key to start")
        self.lbl_hint.setAlignment(Qt.AlignCenter)
        self.lbl_hint.setFont(QFont("Inter", 9))
        self.lbl_hint.setStyleSheet("color: #414C59;")
        lay.addWidget(self.lbl_hint)

        # Opacity animation
        self._eff = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._eff)
        self._anim_in = None
        self._anim_out = None
        self._timer = None
        self._is_dismissed = False

    def start_splash(self, duration_ms: int = 1200):
        self.show()
        self.raise_()
        self._eff.setOpacity(0.0)

        # Fade In
        self._anim_in = QPropertyAnimation(self._eff, b"opacity")
        self._anim_in.setDuration(300)
        self._anim_in.setStartValue(0.0)
        self._anim_in.setEndValue(1.0)
        self._anim_in.setEasingCurve(QEasingCurve.OutCubic)
        self._anim_in.start()

        # Timer to start fade out
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.dismiss)
        self._timer.start(duration_ms)

    def dismiss(self):
        if self._is_dismissed:
            return
        self._is_dismissed = True
        if hasattr(self, "_timer") and self._timer.isActive():
            self._timer.stop()

        self._anim_out = QPropertyAnimation(self._eff, b"opacity")
        self._anim_out.setDuration(250)
        self._anim_out.setStartValue(self._eff.opacity())
        self._anim_out.setEndValue(0.0)
        self._anim_out.setEasingCurve(QEasingCurve.InCubic)
        self._anim_out.finished.connect(self._on_fadeout_done)
        self._anim_out.start()

    def _on_fadeout_done(self):
        self.hide()
        self.dismissed.emit()

    def mousePressEvent(self, event):
        self.dismiss()

    def keyPressEvent(self, event):
        self.dismiss()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

def _load_qss() -> str:
    qss_path = os.path.join(_ROOT, "gui", "style.qss")
    if os.path.isfile(qss_path):
        with open(qss_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Ensure relative icon URLs resolve cleanly in Qt regardless of working directory
            gui_icons_abs = os.path.join(_ROOT, "gui", "icons").replace("\\", "/")
            content = content.replace("url(gui/icons/", f"url({gui_icons_abs}/")
            return content
    return ""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        register_fonts()
        self.setWindowTitle("SIGNAL ANALYZER")
        self.setMinimumSize(980, 680)
        self.setGeometry(80, 80, 1260, 820)
        self.setStyleSheet(_load_qss())
        self.setAcceptDrops(True)

        self._thread = None
        self._worker = None
        self._pipeline_thread: QThread | None = None
        self._pipeline_worker: PipelineTaskWorker | None = None
        self._last_fec_detect_header: str = ""

        # Animation handlers
        self._brand_pulse_anim = None
        self._prog_anim = None
        self._tab_fade_anim = None

        # State tracking
        self._pipeline_stage: str | None = None
        self._current_file_path: str | None = None
        self._current_samples: np.ndarray | None = None
        self._current_fs: int = 1_000_000
        self._current_spec: dict | None = None
        self._current_info: dict | None = None
        self._current_bits: np.ndarray | None = None
        self._current_symbols: np.ndarray | None = None
        self._current_payload_bits: np.ndarray | None = None
        self._last_predicted_mod: str | None = None
        self._suggested_fs: int | None = None
        self._resume_stage: str | None = None
        # Ground-truth reference and per-stage confidence tracking
        self._ground_truth_bits: np.ndarray | None = None
        self._is_synthetic_file: bool = False
        self._pipeline_low_confidence_flags: dict = {}

        self._build_ui()
        self._setup_shortcuts()
        self._setup_tooltips()

        # Start splash hero moment on launch
        self.splash_overlay.setGeometry(self.centralWidget().rect())
        self.splash_overlay.start_splash(duration_ms=1200)

    @staticmethod
    def _create_output_textedit(color: str = "#E4E9F0", height: int = 70) -> QTextEdit:
        te = QTextEdit()
        te.setReadOnly(True)
        te.setFont(QFont("IBM Plex Mono", 10))
        te.setStyleSheet(
            f"color: {color}; background: #0A0E14; border: 1px solid #1E2733; "
            "padding: 4px 6px;"
        )
        te.setFixedHeight(height)
        te.setLineWrapMode(QTextEdit.WidgetWidth)
        te.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        te.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        te.document().setDocumentMargin(2)
        return te

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ==============================================================
        # 1. TOP BAR
        # ==============================================================
        topbar = QFrame()
        topbar.setObjectName("topbar")
        top_lay = QHBoxLayout(topbar)
        top_lay.setContentsMargins(16, 0, 16, 0)
        top_lay.setSpacing(14)

        # Brand
        self.dot_brand = QLabel()
        self.dot_brand.setObjectName("brandDot")
        self.dot_brand.setFixedSize(8, 8)
        self._brand_eff = QGraphicsOpacityEffect(self.dot_brand)
        self.dot_brand.setGraphicsEffect(self._brand_eff)
        self._brand_pulse_anim = QPropertyAnimation(self._brand_eff, b"opacity")
        self._brand_pulse_anim.setDuration(1200)
        self._brand_pulse_anim.setStartValue(1.0)
        self._brand_pulse_anim.setEndValue(0.35)
        self._brand_pulse_anim.setEasingCurve(QEasingCurve.InOutSine)
        self._brand_pulse_anim.setLoopCount(-1)
        self._brand_pulse_anim.start()

        top_lay.addWidget(self.dot_brand)

        self.lbl_brand = QLabel("SIGNAL ANALYZER")
        self.lbl_brand.setObjectName("brandLabel")
        top_lay.addWidget(self.lbl_brand)

        top_lay.addSpacing(16)

        # File metadata line
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setObjectName("fileField")
        self.lbl_file.setFont(QFont("Inter", 10))
        top_lay.addWidget(self.lbl_file, stretch=1)

        # Top Bar Indeterminate Progress Bar for Busy State
        self.top_busy_progress = QProgressBar()
        self.top_busy_progress.setObjectName("topBusyProgress")
        self.top_busy_progress.setRange(0, 0)
        self.top_busy_progress.setFixedWidth(80)
        self.top_busy_progress.setFixedHeight(4)
        self.top_busy_progress.setTextVisible(False)
        self.top_busy_progress.hide()
        top_lay.addWidget(self.top_busy_progress)

        # Pipeline Status Badge
        self.lbl_pipeline_status = QLabel("● Ready")
        self.lbl_pipeline_status.setObjectName("statusBadge")
        self.lbl_pipeline_status.setProperty("statusState", "ready")
        top_lay.addWidget(self.lbl_pipeline_status)

        root.addWidget(topbar)

        # ==============================================================
        # 2. THREE-COLUMN MAIN BODY
        # ==============================================================
        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)

        # --------------------------------------------------------------
        # LEFT COLUMN: Pipeline Stepper (230px)
        # --------------------------------------------------------------
        left_panel = QFrame()
        left_panel.setObjectName("leftPanel")
        left_panel.setFixedWidth(230)
        left_lay = QVBoxLayout(left_panel)
        left_lay.setContentsMargins(16, 16, 16, 16)
        left_lay.setSpacing(12)

        lbl_pipe_head = QLabel("PIPELINE")
        lbl_pipe_head.setObjectName("sectionHeading")
        left_lay.addWidget(lbl_pipe_head)

        # Stepper items
        self.stage_load     = PipelineStageWidget("File Load", "No file loaded")
        self.stage_classify = PipelineStageWidget("Classify", "Pending")
        self.stage_demod    = PipelineStageWidget("Demodulate", "Pending")
        self.stage_sync     = PipelineStageWidget("Sync Detect", "Pending")
        self.stage_fec      = PipelineStageWidget("FEC / Interleave", "Pending")
        self.stage_decode   = PipelineStageWidget("Decode Output", "Pending")

        left_lay.addWidget(self.stage_load)
        left_lay.addWidget(self.stage_classify)
        left_lay.addWidget(self.stage_demod)
        left_lay.addWidget(self.stage_sync)
        left_lay.addWidget(self.stage_fec)
        left_lay.addWidget(self.stage_decode)
        left_lay.addStretch(1)

        body_lay.addWidget(left_panel)

        # --------------------------------------------------------------
        # CENTER COLUMN: Viewport + Stack + Alert Banner + Footer
        # --------------------------------------------------------------
        center_panel = QFrame()
        center_panel.setObjectName("viewportArea")
        center_lay = QVBoxLayout(center_panel)
        center_lay.setContentsMargins(0, 0, 0, 0)
        center_lay.setSpacing(0)

        # Alert Banner at top of viewport
        self.alert_banner = AlertBanner(center_panel)
        center_lay.addWidget(self.alert_banner)

        # Center Viewport Stack: 0 = Empty State, 1 = Plots/Tabs
        self.viewport_stack = QStackedWidget()

        # Page 0: Empty State
        self.empty_state = EmptyStateWidget()
        self.empty_state.open_requested.connect(self._open_file)
        self.viewport_stack.addWidget(self.empty_state)

        # Page 1: Tabbed Viewport
        self.tabs = QTabWidget()
        self.canvas_spec    = MplCanvas(width=8, height=4, dpi=100)
        self.canvas_wfall   = MplCanvas(width=8, height=4, dpi=100)
        self.canvas_constel = MplCanvas(width=5, height=5, dpi=100)

        # Bitstream view
        self.txt_bitstream_view = QTextEdit()
        self.txt_bitstream_view.setReadOnly(True)
        self.txt_bitstream_view.setFont(QFont("IBM Plex Mono", 10))
        self.txt_bitstream_view.setPlaceholderText("Bitstream and demodulated symbols will appear here...")

        self.tabs.addTab(self._wrap(self.canvas_spec),    "Spectrum")
        self.tabs.addTab(self._wrap(self.canvas_wfall),   "Waterfall")
        self.tabs.addTab(self._wrap(self.canvas_constel), "Constellation")
        self.tabs.addTab(self._wrap(self.txt_bitstream_view), "Bitstream")
        self.tabs.setTabToolTip(0, "Spectrum view (shortcut: 1)")
        self.tabs.setTabToolTip(1, "Waterfall view (shortcut: 2)")
        self.tabs.setTabToolTip(2, "Constellation view (shortcut: 3)")
        self.tabs.setTabToolTip(3, "Bitstream view (shortcut: 4)")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.viewport_stack.addWidget(self.tabs)
        self.viewport_stack.setCurrentIndex(0)  # Start at empty state

        center_lay.addWidget(self.viewport_stack, stretch=1)

        # Drop Overlay
        self.drop_overlay = DropOverlay(center_panel)

        # Viewport Footer Readout Bar
        footer = QFrame()
        footer.setObjectName("viewportFooter")
        foot_lay = QHBoxLayout(footer)
        foot_lay.setContentsMargins(16, 0, 16, 0)
        foot_lay.setSpacing(24)

        self.lbl_readout_snr = QLabel("SNR  <b>—</b>")
        self.lbl_readout_snr.setProperty("class", "readout-label")
        self.lbl_readout_cfo = QLabel("CFO  <b>—</b>")
        self.lbl_readout_cfo.setProperty("class", "readout-label")
        self.lbl_readout_phase = QLabel("Phase  <b>—</b>")
        self.lbl_readout_phase.setProperty("class", "readout-label")
        self.lbl_readout_bw = QLabel("OccBW  <b>—</b>")
        self.lbl_readout_bw.setProperty("class", "readout-label")
        self.lbl_readout_samples = QLabel("Samples  <b>—</b>")
        self.lbl_readout_samples.setProperty("class", "readout-label")

        foot_lay.addWidget(self.lbl_readout_snr)
        foot_lay.addWidget(self.lbl_readout_cfo)
        foot_lay.addWidget(self.lbl_readout_phase)
        foot_lay.addWidget(self.lbl_readout_bw)
        foot_lay.addWidget(self.lbl_readout_samples)
        foot_lay.addStretch(1)

        center_lay.addWidget(footer)
        body_lay.addWidget(center_panel, stretch=1)

        # --------------------------------------------------------------
        # RIGHT COLUMN: Metrics + Auto-Detect + Actions (330px)
        # --------------------------------------------------------------
        right_panel = QFrame()
        right_panel.setObjectName("rightPanel")
        right_panel.setFixedWidth(330)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        right_content = QWidget()
        right_lay = QVBoxLayout(right_content)
        right_lay.setContentsMargins(14, 14, 14, 14)
        right_lay.setSpacing(10)

        # Section 1: AUTO-DETECT FEC
        lbl_ad_head = QLabel("AUTO-DETECT FEC")
        lbl_ad_head.setObjectName("sectionHeading")
        right_lay.addWidget(lbl_ad_head)

        card_ad = QFrame()
        card_ad.setObjectName("autodetectCard")
        cad_lay = QVBoxLayout(card_ad)
        cad_lay.setContentsMargins(10, 10, 10, 10)
        cad_lay.setSpacing(6)

        ad_top = QHBoxLayout()
        self.lbl_auto_fec_name = QLabel("Awaiting sync...")
        self.lbl_auto_fec_name.setFont(QFont("IBM Plex Mono", 10, QFont.Bold))
        self.lbl_auto_fec_name.setStyleSheet("color: #4EE1B8;")
        self.lbl_auto_fec_pct = QLabel("—")
        self.lbl_auto_fec_pct.setFont(QFont("IBM Plex Mono", 11, QFont.Bold))
        self.lbl_auto_fec_pct.setStyleSheet("color: #4EE1B8;")
        ad_top.addWidget(self.lbl_auto_fec_name)
        ad_top.addStretch(1)
        ad_top.addWidget(self.lbl_auto_fec_pct)
        cad_lay.addLayout(ad_top)

        self.prog_auto_fec = QProgressBar()
        self.prog_auto_fec.setRange(0, 100)
        self.prog_auto_fec.setValue(0)
        self.prog_auto_fec.setTextVisible(False)
        cad_lay.addWidget(self.prog_auto_fec)

        self.lbl_auto_fec_runner1 = QLabel("Runner-up: —")
        self.lbl_auto_fec_runner1.setFont(QFont("IBM Plex Mono", 9))
        self.lbl_auto_fec_runner1.setStyleSheet("color: #6B7684;")
        cad_lay.addWidget(self.lbl_auto_fec_runner1)

        right_lay.addWidget(card_ad)

        # Section 2: DECODE RESULT METRICS
        lbl_dec_head = QLabel("DECODE RESULT")
        lbl_dec_head.setObjectName("sectionHeading")
        right_lay.addWidget(lbl_dec_head)

        card_metrics = QFrame()
        card_metrics.setObjectName("card")
        cm_lay = QVBoxLayout(card_metrics)
        cm_lay.setContentsMargins(10, 8, 10, 8)
        cm_lay.setSpacing(4)

        self.lbl_m_bits = self._create_metric_row(cm_lay, "Bits recovered", "—", highlight=True)
        self.lbl_m_rate = self._create_metric_row(cm_lay, "Rate", "—")
        self.lbl_m_ber  = self._create_metric_row(cm_lay, "Ground-truth BER", "—", highlight=True)
        self.lbl_m_match= self._create_metric_row(cm_lay, "Re-encode match", "—", highlight=True)

        right_lay.addWidget(card_metrics)

        # Output / details box
        self.lbl_decode = self._create_output_textedit("#55efc4", height=70)
        right_lay.addWidget(self.lbl_decode)

        # Hidden / compatibility labels maintained for existing tests
        self.lbl_class = QLabel("")
        self.lbl_class.setVisible(False)
        right_lay.addWidget(self.lbl_class)

        self.lbl_demod = self._create_output_textedit("#bd93f9", height=40)
        self.lbl_demod.setVisible(False)
        right_lay.addWidget(self.lbl_demod)

        self.lbl_sync = self._create_output_textedit("#ffb86c", height=40)
        self.lbl_sync.setVisible(False)
        right_lay.addWidget(self.lbl_sync)

        self.lbl_info = QLabel("—")
        self.lbl_info.setVisible(False)
        right_lay.addWidget(self.lbl_info)

        # Section 3: ACTIONS & CONTROLS
        lbl_act_head = QLabel("ACTIONS")
        lbl_act_head.setObjectName("sectionHeading")
        right_lay.addWidget(lbl_act_head)

        # Action Buttons
        self.btn_open = QPushButton("Open File (.iq / .wav)")
        self.btn_open.clicked.connect(self._open_file)
        right_lay.addWidget(self.btn_open)

        self.btn_classify = QPushButton("Classify Modulation")
        self.btn_classify.clicked.connect(self._classify)
        self.btn_classify.setEnabled(False)
        right_lay.addWidget(self.btn_classify)

        self.btn_demod = QPushButton("Demodulate")
        self.btn_demod.clicked.connect(self._demodulate)
        self.btn_demod.setEnabled(False)
        right_lay.addWidget(self.btn_demod)

        # Sync controls
        sync_row = QHBoxLayout()
        self.btn_sync = QPushButton("Find Sync")
        self.btn_sync.clicked.connect(self._find_sync)
        self.btn_sync.setEnabled(False)

        self.cmb_sync = QComboBox()
        self.cmb_sync.addItem("Default (0xEB902A3C)", "0xEB902A3C")
        self.cmb_sync.addItem("Alt A (0x1ACFFC1D)", "0x1ACFFC1D")
        self.cmb_sync.addItem("Alt B (0xFAF334BE)", "0xFAF334BE")
        self.cmb_sync.addItem("Alt C (0x352EF853)", "0x352EF853")
        self.cmb_sync.setEditable(True)
        sync_row.addWidget(self.btn_sync, stretch=1)
        sync_row.addWidget(self.cmb_sync, stretch=1)
        right_lay.addLayout(sync_row)

        self.btn_auto_fec = QPushButton("Auto-Detect FEC")
        self.btn_auto_fec.clicked.connect(self._auto_detect_fec)
        self.btn_auto_fec.setEnabled(False)
        right_lay.addWidget(self.btn_auto_fec)

        # FEC scheme picker & RS picker
        self.cmb_fec = QComboBox()
        self.cmb_fec.addItem("Viterbi + Block (8x32)", "viterbi_block")
        self.cmb_fec.addItem("Viterbi + Conv (8x4)", "viterbi_conv")
        self.cmb_fec.addItem("Viterbi + Pseudo-Random Diagonal", "viterbi_diagonal")
        self.cmb_fec.addItem("Viterbi + True Diagonal (16x16)", "viterbi_true_diagonal")
        self.cmb_fec.addItem("Reed-Solomon RS(64,48)", "rs")
        self.cmb_fec.addItem("LDPC (128,64)", "ldpc")
        self.cmb_fec.addItem("Concatenated (RS+Viterbi)", "concatenated")
        self.cmb_fec.currentIndexChanged.connect(self._on_fec_changed)
        right_lay.addWidget(self.cmb_fec)

        self.cmb_rs_size = QComboBox()
        self.cmb_rs_size.addItem("RS(32,24)  — t=4",   (32,  24))
        self.cmb_rs_size.addItem("RS(64,48)  — t=8",   (64,  48))
        self.cmb_rs_size.addItem("RS(128,112) — t=8",  (128, 112))
        self.cmb_rs_size.setCurrentIndex(1)
        self.cmb_rs_size.setVisible(False)
        right_lay.addWidget(self.cmb_rs_size)

        # Primary Decode button
        self.btn_decode = QPushButton("Decode (FEC)")
        self.btn_decode.setObjectName("btnPrimary")
        self.btn_decode.clicked.connect(self._decode)
        self.btn_decode.setEnabled(False)
        right_lay.addWidget(self.btn_decode)

        # Sample rate row
        fs_row = QHBoxLayout()
        self.cmb_fs = QComboBox()
        for label, val in [("Fs = 1 MHz", 1_000_000),
                            ("Fs = 2 MHz", 2_000_000),
                            ("Fs = 250 kHz", 250_000),
                            ("Fs = 48 kHz", 48_000)]:
            self.cmb_fs.addItem(label, val)
        self.cmb_fs.currentIndexChanged.connect(self._on_fs_combo_changed)

        self.btn_apply_fs = QPushButton("Apply")
        self.btn_apply_fs.setEnabled(False)
        self.btn_apply_fs.clicked.connect(self._apply_suggested_fs)

        fs_row.addWidget(self.cmb_fs, stretch=2)
        fs_row.addWidget(self.btn_apply_fs, stretch=1)
        right_lay.addLayout(fs_row)

        self.lbl_rate_est = QLabel("")
        self.lbl_rate_est.setFont(QFont("IBM Plex Mono", 8))
        self.lbl_rate_est.setStyleSheet("color: #F0A84E;")
        right_lay.addWidget(self.lbl_rate_est)

        # Section 4: EXPORT & REPORT
        lbl_exp_head = QLabel("EXPORT & REPORT")
        lbl_exp_head.setObjectName("sectionHeading")
        right_lay.addWidget(lbl_exp_head)

        exp_row = QHBoxLayout()
        self.btn_export = QPushButton("Export Data")
        self.btn_export.clicked.connect(self._export_results)
        self.btn_export.setEnabled(False)

        self.btn_report = QPushButton("Save Report")
        self.btn_report.clicked.connect(self._save_report)
        self.btn_report.setEnabled(False)

        exp_row.addWidget(self.btn_export)
        exp_row.addWidget(self.btn_report)
        right_lay.addLayout(exp_row)

        self.btn_reset = QPushButton("Clear / Reset")
        self.btn_reset.setObjectName("btnGhost")
        self.btn_reset.clicked.connect(self._reset_state)
        right_lay.addWidget(self.btn_reset)

        right_lay.addStretch(1)

        right_scroll.setWidget(right_content)
        r_panel_lay = QVBoxLayout(right_panel)
        r_panel_lay.setContentsMargins(0, 0, 0, 0)
        r_panel_lay.addWidget(right_scroll)

        body_lay.addWidget(right_panel)
        root.addWidget(body, stretch=1)

        # Splash Hero Overlay
        self.splash_overlay = SplashHeroOverlay(central)

        # ==============================================================
        # 3. BOTTOM STATUS BAR
        # ==============================================================
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — open an IQ or WAV file to begin.")

    # ------------------------------------------------------------------
    # Shortcuts & Tooltips
    # ------------------------------------------------------------------

    def _setup_shortcuts(self):
        """Bind application-wide keyboard shortcuts."""
        QShortcut(QKeySequence("Ctrl+O"), self, self._open_file)
        QShortcut(QKeySequence("Ctrl+D"), self, lambda: self._decode() if self.btn_decode.isEnabled() else None)
        QShortcut(QKeySequence("Ctrl+R"), self, self._reset_state)
        QShortcut(QKeySequence("1"), self, lambda: self.tabs.setCurrentIndex(0))
        QShortcut(QKeySequence("2"), self, lambda: self.tabs.setCurrentIndex(1))
        QShortcut(QKeySequence("3"), self, lambda: self.tabs.setCurrentIndex(2))
        QShortcut(QKeySequence("4"), self, lambda: self.tabs.setCurrentIndex(3))
        QShortcut(QKeySequence("Esc"), self, self.hide_banner)

    def _setup_tooltips(self):
        """Set descriptive tooltips across all controls, metrics, and stages."""
        self.btn_open.setToolTip("Open RF signal file (.iq, .wav, .bin) [Ctrl+O]")
        self.btn_classify.setToolTip("Run neural/spectral classifier to identify modulation scheme")
        self.btn_demod.setToolTip("Demodulate baseband samples into symbols and raw bitstream")
        self.btn_sync.setToolTip("Correlate bitstream against 32-bit preamble to locate frame sync")
        self.cmb_sync.setToolTip("Select or type 32-bit sync word in hexadecimal")
        self.btn_auto_fec.setToolTip("Blindly identify FEC code & interleaver from payload consistency")
        self.cmb_fec.setToolTip("Select FEC decoding and de-interleaving scheme")
        self.cmb_rs_size.setToolTip("Select Reed-Solomon (n, k) codeword and payload sizes")
        self.btn_decode.setToolTip("Execute de-interleaving and FEC decode on synchronized payload [Ctrl+D]")
        self.btn_apply_fs.setToolTip("Reload signal using selected sample rate")
        self.btn_export.setToolTip("Export classification, sync, and decoded payload as JSON or CSV")
        self.btn_report.setToolTip("Save analysis report as PNG screenshot or 1-page PDF summary")
        self.btn_reset.setToolTip("Clear all data, reset pipeline stages, and return to empty state [Ctrl+R]")

        self.lbl_readout_snr.setToolTip("Estimated Signal-to-Noise Ratio (dB) via spectral integration")
        self.lbl_readout_cfo.setToolTip("Carrier Frequency Offset (Hz) estimated from spectral center")
        self.lbl_readout_phase.setToolTip("Phase rotation lock state / quadrant ambiguity")
        self.lbl_readout_bw.setToolTip("99% Occupied Bandwidth")
        self.lbl_readout_samples.setToolTip("Total complex sample count in signal buffer")

        self.lbl_m_bits.setToolTip("Number of decoded information bits recovered")
        self.lbl_m_rate.setToolTip("Effective code rate (k/n) of applied FEC scheme")
        self.lbl_m_ber.setToolTip(
            "Bit error rate vs. known reference bits — only shown for synthetic files "
            "that have a companion <basename>_bits.npy file. Shows 'N/A (real capture)' otherwise."
        )
        self.lbl_m_match.setToolTip("Syndrome validation / re-encode parity check match")

        self.stage_load.setToolTip("Stage 1: Load raw IQ or WAV samples into memory")
        self.stage_classify.setToolTip("Stage 2: Detect modulation scheme (BPSK/QPSK/16QAM/2FSK)")
        self.stage_demod.setToolTip("Stage 3: Demodulate baseband symbols into raw bits")
        self.stage_sync.setToolTip("Stage 4: Locate 32-bit preamble and resolve phase rotation")
        self.stage_fec.setToolTip("Stage 5: Identify FEC scheme and de-interleaver parameters")
        self.stage_decode.setToolTip("Stage 6: Recover corrected data payload bits")

    # ------------------------------------------------------------------
    # Drag-and-Drop & Window Handling
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "drop_overlay") and hasattr(self, "viewport_stack"):
            vp_geo = self.viewport_stack.geometry()
            self.drop_overlay.setGeometry(vp_geo)
        if hasattr(self, "splash_overlay") and hasattr(self, "centralWidget") and self.centralWidget():
            self.splash_overlay.setGeometry(self.centralWidget().rect())

    def keyPressEvent(self, event):
        if hasattr(self, "splash_overlay") and self.splash_overlay.isVisible():
            self.splash_overlay.dismiss()
            event.accept()
            return
        super().keyPressEvent(event)

    def _set_drop_overlay_visible(self, visible: bool):
        if hasattr(self, "drop_overlay"):
            if visible:
                vp_geo = self.viewport_stack.geometry()
                self.drop_overlay.setGeometry(vp_geo)
                self.drop_overlay.raise_()
                self.drop_overlay.show()
            else:
                self.drop_overlay.hide()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            valid_exts = (".iq", ".wav", ".bin", ".dat", ".raw")
            if any(u.toLocalFile().lower().endswith(valid_exts) for u in urls):
                event.acceptProposedAction()
                self._set_drop_overlay_visible(True)
                return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._set_drop_overlay_visible(False)

    def dropEvent(self, event: QDropEvent):
        self._set_drop_overlay_visible(False)
        if event.mimeData().hasUrls():
            for u in event.mimeData().urls():
                path = u.toLocalFile()
                if os.path.isfile(path):
                    event.acceptProposedAction()
                    self._load(path)
                    break

    # ------------------------------------------------------------------
    # Banner Notifications
    # ------------------------------------------------------------------

    def show_banner(self, message: str, level: str = "error"):
        """Display dismissible banner with level in ('error', 'warn', 'success')."""
        if hasattr(self, "alert_banner"):
            self.alert_banner.show_message(message, level=level)

    def hide_banner(self):
        """Hide the alert banner."""
        if hasattr(self, "alert_banner"):
            self.alert_banner.dismiss()

    def _set_status_pill(self, text: str, state: str = "ready"):
        """Update the top-right status pill with state in ('ready', 'busy', 'error')."""
        self.lbl_pipeline_status.setText(text)
        self.lbl_pipeline_status.setProperty("statusState", state)
        self.lbl_pipeline_status.style().unpolish(self.lbl_pipeline_status)
        self.lbl_pipeline_status.style().polish(self.lbl_pipeline_status)

    @staticmethod
    def _create_metric_row(parent_layout: QVBoxLayout, label: str, default_val: str = "—", highlight: bool = False) -> QLabel:
        row = QFrame()
        row.setProperty("class", "metric-row")
        rlay = QHBoxLayout(row)
        rlay.setContentsMargins(0, 3, 0, 3)

        klbl = QLabel(label)
        klbl.setProperty("class", "metric-k")
        vlbl = QLabel(default_val)
        vlbl.setProperty("class", "metric-v")
        if highlight:
            vlbl.setProperty("highlight", "true")

        rlay.addWidget(klbl)
        rlay.addStretch(1)
        rlay.addWidget(vlbl)
        parent_layout.addWidget(row)
        return vlbl

    @staticmethod
    def _wrap(widget: QWidget) -> QWidget:
        w = QWidget()
        w.setAutoFillBackground(True)
        w.setAttribute(Qt.WA_OpaquePaintEvent, True)
        w.setStyleSheet("background-color: #0A0E14;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.addWidget(widget)
        return w

    # ------------------------------------------------------------------
    # Button State Management & Async Pipeline Runner
    # ------------------------------------------------------------------

    def _update_button_states(self, busy: bool = False):
        """Enable or disable buttons according to pipeline stage and busy status."""
        if busy:
            self.btn_open.setEnabled(False)
            self.btn_classify.setEnabled(False)
            self.btn_demod.setEnabled(False)
            self.btn_sync.setEnabled(False)
            self.btn_auto_fec.setEnabled(False)
            self.btn_decode.setEnabled(False)
            self.btn_apply_fs.setEnabled(False)
            self.btn_export.setEnabled(False)
            self.btn_report.setEnabled(False)
            self.btn_reset.setEnabled(False)
            if hasattr(self, "top_busy_progress"):
                self.top_busy_progress.show()
        else:
            if hasattr(self, "top_busy_progress"):
                self.top_busy_progress.hide()
            self.btn_open.setEnabled(True)
            self.btn_reset.setEnabled(True)

            stage = getattr(self, "_pipeline_stage", None)
            has_samples = hasattr(self, "_current_samples") and self._current_samples is not None
            has_bits = hasattr(self, "_current_bits") and self._current_bits is not None
            has_payload = hasattr(self, "_current_payload_bits") and self._current_payload_bits is not None and len(self._current_payload_bits) > 0

            self.btn_classify.setEnabled(has_samples)
            self.btn_demod.setEnabled(has_samples and getattr(self, "_last_predicted_mod", None) in ("BPSK", "QPSK", "16QAM", "2FSK"))
            self.btn_sync.setEnabled(has_bits)
            self.btn_auto_fec.setEnabled(has_payload)
            self.btn_decode.setEnabled(has_payload)
            self.btn_export.setEnabled(stage in ("synced", "decoded"))
            self.btn_report.setEnabled(has_samples)

            is_wav = False
            if getattr(self, "_current_file_path", None):
                is_wav = os.path.splitext(self._current_file_path)[1].lower() == ".wav"
            self.cmb_fs.setEnabled(not is_wav)
            self.btn_apply_fs.setEnabled(bool(getattr(self, "_suggested_fs", None)) and not is_wav)

    def _run_pipeline_task(self, task_fn, on_finished, on_error=None, busy_text="Working..."):
        """Execute task_fn in a background QThread, keeping the GUI responsive."""
        self._update_button_states(busy=True)
        self._set_status_pill(f"● {busy_text}", state="busy")
        self.status.showMessage(f"{busy_text} ...")

        thread = PipelineTaskThread(task_fn, parent=self)
        self._pipeline_thread = thread

        def _on_success(res):
            self._update_button_states(busy=False)
            if on_finished:
                on_finished(res)

        def _on_fail(err_msg):
            self._update_button_states(busy=False)
            if on_error:
                on_error(err_msg)
            else:
                self._set_status_pill("● Error", state="error")
                self.show_banner(err_msg, level="error")

        thread.finished_result.connect(_on_success)
        thread.error_occurred.connect(_on_fail)
        thread.start()

    # ------------------------------------------------------------------
    # File open & Load
    # ------------------------------------------------------------------

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Signal File", "",
            "IQ / WAV files (*.iq *.wav *.bin);;All files (*)"
        )
        if not path:
            return
        self._load(path)

    def load_file(self, path: str):
        """Public method so we can call it programmatically."""
        self._load(path)

    def _load(self, path: str):
        self.hide_banner()
        is_wav = os.path.splitext(path)[1].lower() == ".wav"
        self.cmb_fs.setEnabled(not is_wav)
        self.cmb_fs.setToolTip(
            "Ignored — sample rate read from WAV header" if is_wav
            else "Sample rate (for .iq files without embedded rate)"
        )

        self._current_file_path = path
        self._suggested_fs = None

        # Detect companion ground-truth bits file (<basename>_bits.npy)
        bits_companion = os.path.splitext(path)[0] + "_bits.npy"
        if os.path.isfile(bits_companion):
            try:
                self._ground_truth_bits = np.load(bits_companion)
                self._is_synthetic_file = True
                print(f"[Load] Ground-truth bits loaded: {bits_companion} ({len(self._ground_truth_bits)} bits)")
            except Exception as _e:
                self._ground_truth_bits = None
                self._is_synthetic_file = False
                print(f"[Load] Failed to load ground-truth bits: {_e}")
        else:
            self._ground_truth_bits = None
            self._is_synthetic_file = False
        self._pipeline_low_confidence_flags = {}
        fname = os.path.basename(path)
        self.lbl_file.setText(f"Loaded — <b>{fname}</b>")
        self.lbl_rate_est.setText("")
        self.btn_apply_fs.setEnabled(False)
        self.status.showMessage(f"Loading {fname} ...")
        self._set_status_pill("● Loading...", state="busy")

        # Switch viewport stack to plot tabs
        self.viewport_stack.setCurrentIndex(1)

        self._update_button_states(busy=True)

        self.lbl_class.setText("")
        self.lbl_demod.setPlainText("")
        self.lbl_sync.setPlainText("")
        self.lbl_decode.setPlainText("")
        self.txt_bitstream_view.setPlainText("")

        self.stage_load.set_state("active", f"{fname} (loading)")
        self.stage_classify.set_state("pending", "Pending")
        self.stage_demod.set_state("pending", "Pending")
        self.stage_sync.set_state("pending", "Pending")
        self.stage_fec.set_state("pending", "Pending")
        self.stage_decode.set_state("pending", "Pending")
        self._pipeline_stage = None

        fs = self.cmb_fs.currentData()

        self._thread = QThread(self)
        self._worker = AnalysisWorker(path, sample_rate=fs)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    # ------------------------------------------------------------------
    # Pipeline stages: Classify, Demodulate, Sync, FEC Auto-Detect, Decode
    # ------------------------------------------------------------------

    def _classify(self):
        if not hasattr(self, "_current_samples") or self._current_samples is None:
            return
        self.lbl_class.setText("Classifying...")
        self.stage_classify.set_state("active", "Classifying...")
        self.lbl_sync.setPlainText("")
        self.lbl_decode.setPlainText("")

        samples = self._current_samples
        fs = self._current_fs

        def _task():
            from modulation.classifier import predict
            return predict(samples, fs)

        def _on_success(res):
            pred, conf = res
            self.lbl_class.setText(f"Prediction: {pred} ({conf:.1f}% confidence)")
            self._last_predicted_mod = pred
            self._pipeline_stage = "classified"

            self.stage_classify.set_state("done", f"{pred} · {conf:.1f}%")
            self.stage_demod.set_state("active", "Ready to demodulate")
            self._set_status_pill(f"● Classified: {pred}", state="ready")

            # Confidence threshold checks — flag and warn but do not block the pipeline
            if conf < 50.0:
                self._pipeline_low_confidence_flags["classify"] = f"{conf:.1f}%"
                self.show_banner(
                    f"Very low classification confidence ({conf:.1f}%) — result unreliable. "
                    f"Pipeline outputs should not be trusted.",
                    level="error"
                )
            elif conf < 70.0:
                self._pipeline_low_confidence_flags["classify"] = f"{conf:.1f}%"
                self.show_banner(
                    f"Low classification confidence ({conf:.1f}%) — modulation may be "
                    f"misidentified. Verify manually.",
                    level="warn"
                )
            else:
                self._pipeline_low_confidence_flags.pop("classify", None)

            self._update_button_states(busy=False)

        def _on_error(err_msg):
            self.lbl_class.setText(f"Classification error: {err_msg}")
            self.stage_classify.set_state("pending", f"Error: {err_msg}")
            self._set_status_pill("● Classify Error", state="error")
            self.show_banner(f"Modulation classification error: {err_msg}", level="error")
            self._update_button_states(busy=False)

        self._run_pipeline_task(_task, _on_success, _on_error, busy_text="Classifying...")

    def _demodulate(self):
        if not hasattr(self, "_current_samples") or self._current_samples is None:
            return
        self.lbl_demod.setPlainText("Demodulating...")
        self.stage_demod.set_state("active", "Demodulating...")
        self.lbl_sync.setPlainText("")
        self.lbl_decode.setPlainText("")

        samples = self._current_samples
        fs = self._current_fs
        mod = self._last_predicted_mod

        def _task():
            from modulation.demodulator import demodulate
            bits, symbols = demodulate(
                samples, fs, mod, return_symbols=True
            )
            return bits, symbols

        def _on_success(res):
            bits, symbols = res
            self._current_bits = bits
            self._current_symbols = symbols
            self._pipeline_stage = "demodulated"
            bit_str = "".join(str(b) for b in bits)

            self.lbl_demod.setPlainText(f"Demod ({len(bits):,} bits):\n{bit_str}")
            self.txt_bitstream_view.setPlainText(f"--- Demodulated Bitstream ({len(bits):,} bits) ---\n{bit_str}")

            self.stage_demod.set_state("done", f"{len(bits):,} bits recovered")
            self.stage_sync.set_state("active", "Ready for sync detection")
            self._set_status_pill(f"● Demodulated ({len(bits):,} bits)", state="ready")

            fname = os.path.basename(getattr(self, "_current_file_path", "") or "")
            self._plot_constellation(
                symbols, title=f"Constellation (Demodulated {self._last_predicted_mod}) — {fname}"
            )
            self._update_button_states(busy=False)

        def _on_error(err_msg):
            self.lbl_demod.setPlainText(f"Demodulation error: {err_msg}")
            self.stage_demod.set_state("pending", f"Error: {err_msg}")
            self._set_status_pill("● Demod Error", state="error")
            self.show_banner(f"Demodulation error: {err_msg}", level="error")
            self._update_button_states(busy=False)

        self._run_pipeline_task(_task, _on_success, _on_error, busy_text="Demodulating...")

    def _find_sync(self):
        if not hasattr(self, "_current_bits") or self._current_bits is None or len(self._current_bits) == 0:
            return
        self.lbl_sync.setPlainText("Searching for sync word...")
        self.stage_sync.set_state("active", "Searching sync...")
        self.lbl_decode.setPlainText("")

        sync_val = None
        if hasattr(self, "cmb_sync"):
            sync_val = self.cmb_sync.currentData() or self.cmb_sync.currentText()
        bits = self._current_bits
        mod = self._last_predicted_mod

        def _task():
            from modulation.correlator import find_sync, parse_sync_word
            sync_pattern = parse_sync_word(sync_val)
            print(
                f"[SYNC DEBUG] sync_val={sync_val!r} sync_pattern={sync_pattern.tolist()} bits_len={len(bits)} first_20_bits={bits[:20].tolist()}")
            return find_sync(bits, sync_pattern=sync_pattern, modulation_type=mod)

        def _on_success(res):
            payload = res["payload_bits"]
            self._current_payload_bits = payload
            self._pipeline_stage = "synced"
            p_str = "".join(str(b) for b in payload)

            score_pct = res["score"] * 100.0
            sync_name = sync_val if sync_val else "0xEB902A3C"

            # --- Dual confidence gate: correlation score + local signal energy ---
            score_warn_flag = res.get("score_warn", res["score"] < 0.75)
            energy_warn = False
            energy_note = ""
            try:
                _samp = getattr(self, "_current_samples", None)
                _bits = getattr(self, "_current_bits", None)
                if _samp is not None and _bits is not None and len(_bits) > 0:
                    sps = max(1, len(_samp) // len(_bits))
                    s_start = max(0, res["offset"] * sps)
                    window = max(64, 64 * sps)  # ~64 bit-widths of samples
                    s_end = min(len(_samp), s_start + window)
                    local_win = _samp[s_start:s_end]
                    if len(local_win) >= 8:
                        local_pwr = np.mean(np.abs(local_win) ** 2)
                        local_rms_db = 10.0 * np.log10(local_pwr + 1e-12)

                        # Time-domain baseline noise floor & dynamic range
                        block_sz = min(256, max(32, len(_samp) // 100))
                        n_blocks = len(_samp) // block_sz
                        if n_blocks >= 4:
                            blocks = np.abs(_samp[:n_blocks * block_sz].reshape(n_blocks, block_sz)) ** 2
                            block_pwrs_db = 10.0 * np.log10(np.mean(blocks, axis=1) + 1e-12)
                            sorted_pwrs = np.sort(block_pwrs_db)
                            time_floor_db = float(np.median(sorted_pwrs[:max(1, int(0.4 * n_blocks))]))
                            peak_pwr_db = float(np.max(block_pwrs_db))
                            dyn_range = peak_pwr_db - time_floor_db
                        else:
                            time_floor_db = -80.0
                            dyn_range = 0.0

                        # If signal has elevated burst energy (dynamic range >= 6 dB),
                        # candidate lock must sit in active region (local power >= floor + 6 dB)
                        if dyn_range >= 6.0 and local_rms_db < (time_floor_db + 6.0):
                            energy_warn = True
                            energy_note = (
                                f" [NOISE REGION: local {local_rms_db:.1f} dBFS < "
                                f"floor {time_floor_db:.1f} + 6 dB]"
                            )
            except Exception as _e:
                print(f"[SYNC] Energy cross-check error: {_e}")

            # Clear stale flags before re-evaluating
            self._pipeline_low_confidence_flags.pop("sync_energy", None)
            self._pipeline_low_confidence_flags.pop("sync_score", None)

            if energy_warn:
                self._pipeline_low_confidence_flags["sync_energy"] = "noise region lock"
                self.show_banner(
                    f"Sync lock at bit {res['offset']} is in a noise-floor region "
                    f"(local signal power at or below noise floor). This lock is likely "
                    f"spurious — verify against signal timing or try a different sync word.",
                    level="error"
                )
            elif score_warn_flag:
                self._pipeline_low_confidence_flags["sync_score"] = f"{score_pct:.1f}%"
                self.show_banner(
                    f"Low sync correlation score ({score_pct:.1f}%) — sync position "
                    f"uncertain. Verify manually or try a different sync word.",
                    level="warn"
                )

            # Lock status label reflects confidence
            if energy_warn:
                lock_label = "⚡ SUSPECT SYNC (NOISE REGION)"
            elif score_warn_flag:
                lock_label = "⚠ SYNC LOCKED (LOW SCORE)"
            else:
                lock_label = "SYNC LOCKED"

            self.lbl_sync.setPlainText(
                f"Sync @ bit {res['offset']} ({score_pct:.1f}% match, {res['variant']}) "
                f"using {sync_name}{energy_note} | Payload ({len(payload):,} bits):\n{p_str}"
            )

            current_b_text = self.txt_bitstream_view.toPlainText()
            self.txt_bitstream_view.setPlainText(
                f"[{lock_label}: @ bit {res['offset']} ({score_pct:.1f}% match, "
                f"{res['variant']}) using {sync_name}]{energy_note}\n"
                f"Payload: {len(payload)} bits\n"
                f"{p_str}\n\n"
                f"{current_b_text}"
            )

            suffix = " ⚡" if energy_warn else (" ⚠" if score_warn_flag else "")
            self.stage_sync.set_state(
                "done",
                f"{score_pct:.0f}% match · bit {res['offset']} ({sync_name}){suffix}"
            )
            self.stage_fec.set_state("active", "Ready for FEC auto-detect")
            if energy_warn:
                pill_state = "error"
                pill_text = f"⚡ Suspect Sync (Noise Region @ bit {res['offset']})"
            elif score_warn_flag:
                pill_state = "warn"
                pill_text = f"⚠ Sync Locked (Low Score @ bit {res['offset']})"
            else:
                pill_state = "ready"
                pill_text = f"● Sync Locked @ bit {res['offset']} ({sync_name})"
            self._set_status_pill(pill_text, state=pill_state)

            self.lbl_readout_phase.setText(f"Phase  <b>{res['variant']}</b>")
            self._update_button_states(busy=False)

        def _on_error(err_msg):
            self.lbl_sync.setPlainText(f"Sync error: {err_msg}")
            self.stage_sync.set_state("pending", f"Error: {err_msg}")
            self._set_status_pill("● Sync Error", state="error")
            self.show_banner(f"Sync detection error: {err_msg}", level="error")
            self._update_button_states(busy=False)

        self._run_pipeline_task(_task, _on_success, _on_error, busy_text="Finding Sync...")

    def _auto_detect_fec(self):
        """Blindly detect the FEC / interleaving scheme from synchronized payload bits."""
        if not hasattr(self, "_current_payload_bits") or self._current_payload_bits is None or len(self._current_payload_bits) == 0:
            self.lbl_decode.setPlainText("Auto-Detect FEC error: No synchronized payload bits available. Run 'Find Sync' first.")
            self.show_banner("Cannot detect FEC: No synchronized payload bits. Run 'Find Sync' first.", level="warn")
            return

        self.lbl_decode.setPlainText("Blindly identifying FEC / interleaving scheme...")
        self.stage_fec.set_state("active", "Detecting...")

        payload = self._current_payload_bits

        def _task():
            from modulation.fec_identifier import identify_fec_scheme
            return identify_fec_scheme(payload)

        def _on_success(results):
            if not results:
                self.lbl_decode.setPlainText(
                    f"Auto-Detect FEC: No matching FEC scheme identified for payload of {len(payload)} bits."
                )
                self.lbl_auto_fec_name.setText("No Match")
                self.lbl_auto_fec_pct.setText("0%")
                self.prog_auto_fec.setValue(0)
                self.show_banner(f"No consistent FEC pattern identified for {len(payload)} payload bits.", level="warn")
                self._update_button_states(busy=False)
                return

            top = results[0]
            self.lbl_auto_fec_name.setText(top.name)
            self.lbl_auto_fec_pct.setText(f"{top.confidence*100:.0f}%")
            
            # Smoothly animate progress bar fill
            target_prog = int(top.confidence * 100)
            self._prog_anim = QPropertyAnimation(self.prog_auto_fec, b"value")
            self._prog_anim.setDuration(500)
            self._prog_anim.setStartValue(self.prog_auto_fec.value())
            self._prog_anim.setEndValue(target_prog)
            self._prog_anim.setEasingCurve(QEasingCurve.OutCubic)
            self._prog_anim.start()

            runner_up_str = f"Runner-up: {results[1].name} ({results[1].confidence*100:.1f}%)" if len(results) > 1 else "No secondary match"
            self.lbl_auto_fec_runner1.setText(runner_up_str)

            # Auto-update cmb_fec dropdown
            idx = -1
            for i in range(self.cmb_fec.count()):
                if self.cmb_fec.itemData(i) == top.scheme_key:
                    idx = i
                    break
            if idx != -1:
                self.cmb_fec.setCurrentIndex(idx)

            if top.scheme_key == "rs" and top.rs_size is not None:
                rs_idx = -1
                for i in range(self.cmb_rs_size.count()):
                    if self.cmb_rs_size.itemData(i) == top.rs_size:
                        rs_idx = i
                        break
                if rs_idx != -1:
                    self.cmb_rs_size.setCurrentIndex(rs_idx)

            self.stage_fec.set_state("done", f"{top.name} ({top.confidence*100:.0f}%)")
            self.stage_decode.set_state("active", "Ready to decode")

            # FEC confidence threshold
            fec_conf_pct = top.confidence * 100.0
            if fec_conf_pct < 70.0:
                self._pipeline_low_confidence_flags["fec"] = f"{fec_conf_pct:.1f}%"
                self.show_banner(
                    f"Low FEC detection confidence ({fec_conf_pct:.1f}%) — scheme "
                    f"identification uncertain. Consider manual FEC selection.",
                    level="warn"
                )
            else:
                self._pipeline_low_confidence_flags.pop("fec", None)

            self._last_fec_detect_header = (
                f"[Auto-Detected FEC: {top.name} ({top.confidence*100:.1f}% confidence)]\n"
                f"Consistency details: {top.details}\n"
                f"{'='*60}\n"
            )

            # Trigger decode with the identified scheme
            self._decode()

        def _on_error(err_msg):
            self.lbl_decode.setPlainText(f"Auto-Detect FEC error: {err_msg}")
            self.stage_fec.set_state("pending", f"Error: {err_msg}")
            self._set_status_pill("● FEC Error", state="error")
            self.show_banner(f"Auto-Detect FEC error: {err_msg}", level="error")
            self._update_button_states(busy=False)

        self._run_pipeline_task(_task, _on_success, _on_error, busy_text="Identifying FEC...")

    def _on_fec_changed(self, _index: int):
        """Show the RS size picker only when Reed-Solomon is the active FEC scheme."""
        is_rs = (self.cmb_fec.currentData() == "rs")
        self.cmb_rs_size.setVisible(is_rs)

    def _decode(self):
        if not hasattr(self, "_current_payload_bits") or self._current_payload_bits is None or len(self._current_payload_bits) == 0:
            self.lbl_decode.setPlainText("Decode error: No synchronized payload bits available.")
            self.stage_decode.set_state("pending", "No payload bits")
            self.show_banner("Cannot decode: No synchronized payload bits available. Run 'Find Sync' first.", level="warn")
            return

        self.stage_decode.set_state("active", "Decoding...")
        scheme = self.cmb_fec.currentData() if hasattr(self, "cmb_fec") else "viterbi_block"
        payload = self._current_payload_bits
        rs_size = self.cmb_rs_size.currentData() if hasattr(self, "cmb_rs_size") else (64, 48)

        def _task():
            if scheme == "rs":
                from modulation.rs_fec import rs_decode, rs_encode
                n_sym, k_sym = rs_size or (64, 48)
                cw_len = n_sym * 8
                if len(payload) >= cw_len:
                    cw_bits = payload[:cw_len]
                else:
                    cw_bits = np.pad(payload, (0, cw_len - len(payload)))

                cw_bytes = np.packbits(cw_bits)
                dec_bytes = rs_decode(cw_bytes, n=n_sym, k=k_sym)
                re_cw = rs_encode(dec_bytes, n=n_sym, k=k_sym)
                n_err = int(np.sum(cw_bytes != re_cw))
                decoded = np.unpackbits(dec_bytes)
                # Byte-level re-encode match: proportion of codeword bytes that round-trip cleanly
                match_pct = max(0.0, (1.0 - n_err / max(n_sym, 1)) * 100.0)
                return {
                    "scheme": "rs",
                    "decoded": decoded,
                    "n_err": n_err,
                    "n_sym": n_sym,
                    "k_sym": k_sym,
                    "match_pct": match_pct,
                }

            elif scheme == "ldpc":
                from modulation.ldpc import ldpc_decode, ldpc_encode, DEFAULT_H_128_64
                H = DEFAULT_H_128_64
                n = H.shape[1]
                k = n - H.shape[0]

                if len(payload) >= n:
                    cw_bits = payload[:n]
                else:
                    cw_bits = np.pad(payload, (0, n - len(payload)))

                decoded, success, iters = ldpc_decode(cw_bits, H=H, max_iterations=50)
                # Re-encode decoded info bits and compare to received codeword
                try:
                    re_cw = ldpc_encode(decoded).astype(int)
                    comp_n = min(len(re_cw), len(cw_bits))
                    hamming_d = int(np.sum(re_cw[:comp_n] != cw_bits[:comp_n].astype(int)))
                    match_pct = max(0.0, (1.0 - hamming_d / max(comp_n, 1)) * 100.0)
                except Exception:
                    match_pct = None
                return {
                    "scheme": "ldpc",
                    "decoded": decoded,
                    "success": success,
                    "iters": iters,
                    "n": n,
                    "k": k,
                    "match_pct": match_pct,
                }

            elif scheme == "concatenated":
                from modulation.concatenated import concatenated_decode, concatenated_encode
                min_len = 2 * (64 * 8 + 6)
                if len(payload) >= min_len:
                    cw_bits = payload[:min_len]
                else:
                    cw_bits = np.pad(payload, (0, min_len - len(payload)))

                decoded = concatenated_decode(cw_bits, n=64, k=48)
                # Re-encode: pack first k_sym=48 bytes of decoded bits, encode, compare
                try:
                    info_bytes = np.packbits(decoded[:48 * 8])
                    re_cw = concatenated_encode(info_bytes, n=64, k=48)
                    comp_n = min(len(re_cw), len(cw_bits))
                    hamming_d = int(np.sum(re_cw[:comp_n].astype(int) != cw_bits[:comp_n].astype(int)))
                    match_pct = max(0.0, (1.0 - hamming_d / max(comp_n, 1)) * 100.0)
                except Exception:
                    match_pct = None
                return {
                    "scheme": "concatenated",
                    "decoded": decoded,
                    "match_pct": match_pct,
                }

            elif scheme == "viterbi_conv":
                from modulation.deinterleaver import conv_deinterleave
                from modulation.fec import viterbi_decode, conv_encode

                deinterleaved = conv_deinterleave(payload, depth=8, span=4, trim_delay=True)
                decoded = viterbi_decode(deinterleaved)
                # Re-encode decoded → compare to de-interleaved stream (independent roundtrip check)
                try:
                    re_enc = conv_encode(decoded, add_tail=True)
                    comp_n = min(len(re_enc), len(deinterleaved))
                    hamming_d = int(np.sum(re_enc[:comp_n] != deinterleaved[:comp_n]))
                    match_pct = max(0.0, (1.0 - hamming_d / max(comp_n, 1)) * 100.0)
                except Exception:
                    match_pct = None
                return {
                    "scheme": "viterbi_conv",
                    "decoded": decoded,
                    "match_pct": match_pct,
                }

            elif scheme == "viterbi_diagonal":
                from modulation.deinterleaver import diagonal_deinterleave
                from modulation.fec import viterbi_decode, conv_encode

                deinterleaved = diagonal_deinterleave(payload)
                decoded = viterbi_decode(deinterleaved)
                try:
                    re_enc = conv_encode(decoded, add_tail=True)
                    comp_n = min(len(re_enc), len(deinterleaved))
                    hamming_d = int(np.sum(re_enc[:comp_n] != deinterleaved[:comp_n]))
                    match_pct = max(0.0, (1.0 - hamming_d / max(comp_n, 1)) * 100.0)
                except Exception:
                    match_pct = None
                return {
                    "scheme": "viterbi_diagonal",
                    "decoded": decoded,
                    "match_pct": match_pct,
                }

            elif scheme == "viterbi_true_diagonal":
                from modulation.deinterleaver import true_diagonal_deinterleave
                from modulation.fec import viterbi_decode, conv_encode

                deinterleaved = true_diagonal_deinterleave(payload, rows=16, cols=16)
                decoded = viterbi_decode(deinterleaved)
                try:
                    re_enc = conv_encode(decoded, add_tail=True)
                    comp_n = min(len(re_enc), len(deinterleaved))
                    hamming_d = int(np.sum(re_enc[:comp_n] != deinterleaved[:comp_n]))
                    match_pct = max(0.0, (1.0 - hamming_d / max(comp_n, 1)) * 100.0)
                except Exception:
                    match_pct = None
                return {
                    "scheme": "viterbi_true_diagonal",
                    "decoded": decoded,
                    "match_pct": match_pct,
                }

            else:  # "viterbi_block"
                from modulation.deinterleaver import block_deinterleave
                from modulation.fec import viterbi_decode, conv_encode

                deinterleaved = block_deinterleave(payload, rows=8, cols=32)
                decoded = viterbi_decode(deinterleaved)
                try:
                    re_enc = conv_encode(decoded, add_tail=True)
                    comp_n = min(len(re_enc), len(deinterleaved))
                    hamming_d = int(np.sum(re_enc[:comp_n] != deinterleaved[:comp_n]))
                    match_pct = max(0.0, (1.0 - hamming_d / max(comp_n, 1)) * 100.0)
                except Exception:
                    match_pct = None
                return {
                    "scheme": "viterbi_block",
                    "decoded": decoded,
                    "match_pct": match_pct,
                }

        def _on_success(res):
            scheme_type = res["scheme"]
            decoded = res["decoded"]
            bit_str = "".join(str(b) for b in decoded)
            match_pct = res.get("match_pct")

            # --- Ground-truth BER: real only for synthetic files with companion _bits.npy ---
            gt_bits = getattr(self, "_ground_truth_bits", None)
            if gt_bits is not None and len(decoded) > 0:
                comp_n = min(len(decoded), len(gt_bits))
                hamming_ber = int(np.sum(decoded[:comp_n] != gt_bits[:comp_n]))
                ber_str = f"{hamming_ber / max(comp_n, 1) * 100.0:.2f}%"
            else:
                ber_str = "N/A (real capture)"

            # --- Re-encode match: real round-trip percentage from task ---
            match_str = f"{match_pct:.1f}%" if match_pct is not None else "N/A"

            header_title = "[DECODED PAYLOAD]"
            if scheme_type == "rs":
                n_sym = res["n_sym"]
                k_sym = res["k_sym"]
                n_err = res["n_err"]
                err_str = f"corrected {n_err} byte errors" if n_err > 0 else "0 errors detected"
                out_msg = f"Decoded ({len(decoded):,} bits / {k_sym} bytes [RS({n_sym},{k_sym}) {err_str}]):\n{bit_str}"
                header_title = f"[DECODED PAYLOAD: Reed-Solomon RS({n_sym},{k_sym}) | {len(decoded):,} bits / {k_sym} bytes | {err_str}]"
                self.lbl_m_bits.setText(f"{len(decoded)}")
                self.lbl_m_rate.setText(f"{k_sym}/{n_sym}")
                self.lbl_m_ber.setText(ber_str)
                self.lbl_m_match.setText(match_str)

            elif scheme_type == "ldpc":
                n = res["n"]
                k = res["k"]
                iters = res["iters"]
                if res["success"]:
                    out_msg = f"Decoded ({len(decoded):,} bits [LDPC(128,64) rate 1/2, converged in {iters} iters]):\n{bit_str}"
                    header_title = f"[DECODED PAYLOAD: LDPC({n},{k}) Rate 1/2 | {len(decoded):,} bits | Converged in {iters} iters]"
                    self.lbl_m_bits.setText(f"{len(decoded)}")
                    self.lbl_m_rate.setText(f"{k}/{n}")
                    self.lbl_m_ber.setText(ber_str)
                    self.lbl_m_match.setText(match_str)
                else:
                    out_msg = f"LDPC Decode uncorrectable error: parity checks not satisfied after {iters} iterations."
                    header_title = f"[DECODED PAYLOAD: LDPC({n},{k}) | FAILED TO CONVERGE ({iters} iters)]"
                    self.lbl_m_match.setText(f"Failed ({iters} iters)")
                    self.show_banner(f"LDPC decode uncorrectable error: parity checks not satisfied after {iters} iterations.", level="error")

            elif scheme_type == "concatenated":
                out_msg = f"Decoded ({len(decoded):,} bits / {len(decoded)//8} bytes [Concatenated RS(64,48)+Viterbi]):\n{bit_str}"
                header_title = f"[DECODED PAYLOAD: Concatenated RS(64,48)+Viterbi | {len(decoded):,} bits / {len(decoded)//8} bytes]"
                self.lbl_m_bits.setText(f"{len(decoded)}")
                self.lbl_m_rate.setText("48/128")
                self.lbl_m_ber.setText(ber_str)
                self.lbl_m_match.setText(match_str)

            elif scheme_type == "viterbi_conv":
                info_note = " (first 512 bits = info payload)" if len(decoded) >= 512 else ""
                out_msg = f"Decoded ({len(decoded):,} bits [Rate 1/2 Conv]{info_note}):\n{bit_str}"
                header_title = f"[DECODED PAYLOAD: Rate 1/2 Convolutional (Conv Interleaver 8x4) | {len(decoded):,} bits{info_note}]"
                self.lbl_m_bits.setText(f"{len(decoded)}")
                self.lbl_m_rate.setText("1/2")
                self.lbl_m_ber.setText(ber_str)
                self.lbl_m_match.setText(match_str)

            elif scheme_type == "viterbi_diagonal":
                info_note = " (first 512 bits = info payload)" if len(decoded) >= 512 else ""
                out_msg = f"Decoded ({len(decoded):,} bits [Rate 1/2 Pseudo-Random Diagonal]{info_note}):\n{bit_str}"
                header_title = f"[DECODED PAYLOAD: Rate 1/2 Convolutional (Pseudo-Random Diagonal) | {len(decoded):,} bits{info_note}]"
                self.lbl_m_bits.setText(f"{len(decoded)}")
                self.lbl_m_rate.setText("1/2")
                self.lbl_m_ber.setText(ber_str)
                self.lbl_m_match.setText(match_str)

            elif scheme_type == "viterbi_true_diagonal":
                info_note = " (first 512 bits = info payload)" if len(decoded) >= 512 else ""
                out_msg = f"Decoded ({len(decoded):,} bits [Rate 1/2 True Diagonal 16x16]{info_note}):\n{bit_str}"
                header_title = f"[DECODED PAYLOAD: Rate 1/2 Convolutional (True Diagonal 16x16) | {len(decoded):,} bits{info_note}]"
                self.lbl_m_bits.setText(f"{len(decoded)}")
                self.lbl_m_rate.setText("1/2")
                self.lbl_m_ber.setText(ber_str)
                self.lbl_m_match.setText(match_str)

            else:  # "viterbi_block"
                info_note = " (first 512 bits = info payload)" if len(decoded) >= 512 else ""
                out_msg = f"Decoded ({len(decoded):,} bits [Rate 1/2 Block]{info_note}):\n{bit_str}"
                header_title = f"[DECODED PAYLOAD: Rate 1/2 Convolutional (Block Interleaver 8x32) | {len(decoded):,} bits{info_note}]"
                self.lbl_m_bits.setText(f"{len(decoded)}")
                self.lbl_m_rate.setText("1/2")
                self.lbl_m_ber.setText(ber_str)
                self.lbl_m_match.setText(match_str)

            # Prepend a ⚠ warning header to the bitstream block when any stage had low confidence
            flags = getattr(self, "_pipeline_low_confidence_flags", {})
            if flags:
                flag_detail = " | ".join(f"{k}: {v}" for k, v in flags.items())
                header_title = f"[\u26a0 LOW CONFIDENCE \u2014 {flag_detail}]\n{header_title}"

            fec_hdr = getattr(self, "_last_fec_detect_header", "")
            self._last_fec_detect_header = ""
            self.lbl_decode.setPlainText(fec_hdr + out_msg)

            # Prepend decoded payload to Bitstream tab view and switch to it
            current_b_text = self.txt_bitstream_view.toPlainText()
            self.txt_bitstream_view.setPlainText(
                f"{header_title}\n"
                f"{bit_str}\n\n"
                f"{current_b_text}"
            )
            self.tabs.setCurrentIndex(3)

            self._pipeline_stage = "decoded"
            self.stage_decode.set_state("done", f"{len(decoded)} bits recovered")

            # Final pill reflects accumulated pipeline confidence with distinct severity
            if "sync_energy" in flags:
                self._set_status_pill("⚡ Pipeline complete (spurious noise lock)", state="error")
            elif flags:
                self._set_status_pill("⚠ Pipeline complete (low confidence)", state="warn")
            else:
                self._set_status_pill("\u25cf Pipeline complete", state="ready")
            self._update_button_states(busy=False)

        def _on_error(err_msg):
            self.lbl_decode.setPlainText(f"Decode error: {err_msg}")
            self.stage_decode.set_state("pending", f"Error: {err_msg}")
            self._set_status_pill("● Decode Error", state="error")
            self.show_banner(f"Decode error: {err_msg}", level="error")
            self._update_button_states(busy=False)

        self._run_pipeline_task(_task, _on_success, _on_error, busy_text="Decoding...")

    def _on_tab_changed(self, index: int):
        """Exclusively display active tab and cleanly fade in without layer bleed."""
        for i in range(self.tabs.count()):
            page = self.tabs.widget(i)
            if page is not None and i != index:
                page.setGraphicsEffect(None)
                page.setVisible(False)

        active_page = self.tabs.widget(index)
        if active_page is not None:
            active_page.setVisible(True)
            eff = QGraphicsOpacityEffect(active_page)
            active_page.setGraphicsEffect(eff)
            self._tab_fade_anim = QPropertyAnimation(eff, b"opacity")
            self._tab_fade_anim.setDuration(150)
            self._tab_fade_anim.setStartValue(0.4)
            self._tab_fade_anim.setEndValue(1.0)
            self._tab_fade_anim.setEasingCurve(QEasingCurve.OutCubic)
            # Remove the graphics effect completely when finished to prevent raster compositing overlap
            self._tab_fade_anim.finished.connect(lambda p=active_page: p.setGraphicsEffect(None))
            self._tab_fade_anim.start()

            # Trigger redraw on active canvas
            if index == 0 and hasattr(self, "canvas_spec"):
                self.canvas_spec.draw_idle()
            elif index == 1 and hasattr(self, "canvas_wfall"):
                self.canvas_wfall.draw_idle()
            elif index == 2 and hasattr(self, "canvas_constel"):
                self.canvas_constel.draw_idle()

    def _on_fs_combo_changed(self, _index: int):
        """Enable Apply whenever the dropdown changes and a non-WAV file is loaded."""
        path = getattr(self, "_current_file_path", None)
        if path and os.path.splitext(path)[1].lower() != ".wav":
            self.btn_apply_fs.setEnabled(True)
            self.btn_apply_fs.setToolTip(
                f"Reload signal using selected Fs = {self.cmb_fs.currentData()/1e3:.0f} kHz"
            )

    def _apply_suggested_fs(self):
        path = getattr(self, "_current_file_path", None)
        if not path:
            return
        chosen_fs = self.cmb_fs.currentData()
        print(f"[Apply] Reloading with Fs = {chosen_fs/1e3:.0f} kHz")
        self._resume_stage = getattr(self, "_pipeline_stage", None)
        self._load(path)

    # ------------------------------------------------------------------
    # Export & Report Handlers
    # ------------------------------------------------------------------

    def _export_results(self):
        """Export classification, sync score, FEC scheme, BER, and decoded bits as JSON or CSV."""
        if not hasattr(self, "_current_samples") or self._current_samples is None:
            self.show_banner("Cannot export — no signal file loaded.", level="warn")
            return

        default_name = "signal_analysis_export.json"
        if getattr(self, "_current_file_path", None):
            base = os.path.splitext(os.path.basename(self._current_file_path))[0]
            default_name = f"{base}_export.json"

        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Analysis Results", default_name,
            "JSON (*.json);;CSV (*.csv);;All Files (*)"
        )
        if not path:
            return

        export_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "file": {
                "path": getattr(self, "_current_file_path", ""),
                "sample_rate": getattr(self, "_current_fs", 0),
                "sample_count": len(self._current_samples) if hasattr(self, "_current_samples") else 0,
            },
            "metrics": {
                "snr": self.lbl_readout_snr.text().replace("SNR", "").strip(),
                "cfo": self.lbl_readout_cfo.text().replace("CFO", "").strip(),
                "phase": self.lbl_readout_phase.text().replace("Phase", "").strip(),
                "occupied_bw": self.lbl_readout_bw.text().replace("OccBW", "").strip(),
            },
            "classification": {
                "modulation": getattr(self, "_last_predicted_mod", "Unknown"),
            },
            "synchronization": {
                "sync_pattern": self.cmb_sync.currentText() if hasattr(self, "cmb_sync") else "",
                "status": self.stage_sync.lbl_sub.text(),
            },
            "fec_decode": {
                "scheme": self.cmb_fec.currentText() if hasattr(self, "cmb_fec") else "",
                "bits_recovered": self.lbl_m_bits.text(),
                "code_rate": self.lbl_m_rate.text(),
                "ber": self.lbl_m_ber.text(),
                "parity_match": self.lbl_m_match.text(),
                "details": self.lbl_decode.toPlainText(),
            }
        }

        if path.lower().endswith(".csv") or "CSV" in selected_filter:
            if not path.lower().endswith(".csv"):
                path += ".csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Category", "Parameter", "Value"])
                writer.writerow(["Metadata", "Timestamp", export_data["timestamp"]])
                writer.writerow(["Metadata", "File Path", export_data["file"]["path"]])
                writer.writerow(["Metadata", "Sample Rate (Hz)", export_data["file"]["sample_rate"]])
                writer.writerow(["Metadata", "Sample Count", export_data["file"]["sample_count"]])
                writer.writerow(["RF Metrics", "SNR", export_data["metrics"]["snr"]])
                writer.writerow(["RF Metrics", "CFO", export_data["metrics"]["cfo"]])
                writer.writerow(["RF Metrics", "Occupied BW", export_data["metrics"]["occupied_bw"]])
                writer.writerow(["Modulation", "Predicted", export_data["classification"]["modulation"]])
                writer.writerow(["Sync", "Pattern", export_data["synchronization"]["sync_pattern"]])
                writer.writerow(["Sync", "Status", export_data["synchronization"]["status"]])
                writer.writerow(["FEC", "Scheme", export_data["fec_decode"]["scheme"]])
                writer.writerow(["FEC", "Bits Recovered", export_data["fec_decode"]["bits_recovered"]])
                writer.writerow(["FEC", "Rate", export_data["fec_decode"]["code_rate"]])
                writer.writerow(["FEC", "BER", export_data["fec_decode"]["ber"]])
                writer.writerow(["FEC", "Parity Match", export_data["fec_decode"]["parity_match"]])
        else:
            if not path.lower().endswith(".json"):
                path += ".json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)

        self.show_banner(f"Results exported successfully to {os.path.basename(path)}", level="success")
        self.status.showMessage(f"Exported: {path}")

    def _save_report(self):
        """Save graphical report as PNG screenshot or 1-page PDF summary."""
        if not hasattr(self, "_current_samples") or self._current_samples is None:
            self.show_banner("Cannot save report — no signal file loaded.", level="warn")
            return

        default_name = "signal_report.png"
        if getattr(self, "_current_file_path", None):
            base = os.path.splitext(os.path.basename(self._current_file_path))[0]
            default_name = f"{base}_report.png"

        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Analysis Report", default_name,
            "PNG Image (*.png);;PDF Report (*.pdf);;All Files (*)"
        )
        if not path:
            return

        if path.lower().endswith(".pdf") or "PDF" in selected_filter:
            if not path.lower().endswith(".pdf"):
                path += ".pdf"
            self._generate_pdf_report(path)
        else:
            if not path.lower().endswith(".png"):
                path += ".png"
            pixmap = self.grab()
            pixmap.save(path, "PNG")

        self.show_banner(f"Report saved successfully to {os.path.basename(path)}", level="success")
        self.status.showMessage(f"Saved Report: {path}")

    def _generate_pdf_report(self, path: str):
        """Generate a clean 1-page PDF summary report using Matplotlib."""
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        fname = os.path.basename(getattr(self, "_current_file_path", "Signal"))
        fs = getattr(self, "_current_fs", 1_000_000)
        mod = getattr(self, "_last_predicted_mod", "N/A")
        fec = self.cmb_fec.currentText() if hasattr(self, "cmb_fec") else "N/A"
        bits_rec = self.lbl_m_bits.text()
        ber = self.lbl_m_ber.text()

        with PdfPages(path) as pdf:
            fig = plt.figure(figsize=(8.5, 11), facecolor="#0A0E14")
            
            # Title
            fig.text(0.08, 0.94, "SIGNAL ANALYZER — INSPECTION REPORT", fontsize=15, fontweight="bold", color="#4EE1B8")
            fig.text(0.08, 0.915, f"File: {fname}   |   Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=9, color="#6B7684")

            # Metrics Summary Box (Table)
            table_data = [
                ["Sample Rate", f"{fs/1e6:.3f} MHz", "Modulation", mod],
                ["Samples", f"{len(self._current_samples):,}" if hasattr(self, "_current_samples") else "0", "FEC Scheme", fec],
                ["SNR Estimate", self.lbl_readout_snr.text().replace("SNR", "").replace("<b>","").replace("</b>","").strip(), "Bits Recovered", bits_rec],
                ["Carrier Offset", self.lbl_readout_cfo.text().replace("CFO", "").replace("<b>","").replace("</b>","").strip(), "BER", ber],
            ]
            ax_table = fig.add_axes([0.08, 0.77, 0.84, 0.11])
            ax_table.axis("off")
            tbl = ax_table.table(cellText=table_data, colWidths=[0.20, 0.30, 0.20, 0.30], loc="center", cellLoc="left")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8.5)
            for key, cell in tbl.get_celld().items():
                cell.set_edgecolor("#1E2733")
                cell.set_facecolor("#141B25" if key[1] in (0, 2) else "#10151D")
                cell.set_text_props(color="#4EE1B8" if key[1] in (1, 3) else "#E4E9F0")
                cell.set_height(0.24)

            # Subplot 1: Power Spectrum
            ax_spec = fig.add_axes([0.08, 0.44, 0.84, 0.28], facecolor="#10151D")
            if hasattr(self, "_current_samples") and self._current_samples is not None:
                spec = spec_analyse(self._current_samples, fs, nfft=4096)
                kHz = spec["freqs"] / 1e3
                ax_spec.plot(kHz, spec["psd_db"], color="#4EE1B8", lw=1.0)
                ax_spec.fill_between(kHz, spec["psd_db"], spec["psd_db"].min(), alpha=0.1, color="#4EE1B8")
                ax_spec.set_title("Power Spectral Density (PSD)", fontsize=10, color="#E4E9F0", pad=6)
                ax_spec.set_xlabel("Frequency (kHz)", fontsize=8, color="#6B7684")
                ax_spec.set_ylabel("Power (dBFS)", fontsize=8, color="#6B7684")
                ax_spec.tick_params(colors="#6B7684", labelsize=8)
                ax_spec.grid(True, color="#1E2733", linestyle=":", lw=0.6)

            # Subplot 2: Constellation
            ax_cs = fig.add_axes([0.08, 0.10, 0.38, 0.28], facecolor="#10151D")
            if hasattr(self, "_current_symbols") and self._current_symbols is not None:
                cs = self._current_symbols
                pwr = np.sqrt(np.mean(np.abs(cs)**2)) or 1.0
                cs_n = cs / pwr
                ax_cs.scatter(cs_n.real[:5000], cs_n.imag[:5000], s=8, color="#4EE1B8", alpha=0.4)
                ax_cs.set_title("Constellation Diagram", fontsize=10, color="#E4E9F0", pad=6)
                ax_cs.set_xlabel("In-Phase (I)", fontsize=8, color="#6B7684")
                ax_cs.set_ylabel("Quadrature (Q)", fontsize=8, color="#6B7684")
                ax_cs.tick_params(colors="#6B7684", labelsize=8)
                ax_cs.set_xlim(-1.5, 1.5)
                ax_cs.set_ylim(-1.5, 1.5)
                ax_cs.grid(True, color="#1E2733", linestyle=":", lw=0.6)
            else:
                ax_cs.text(0.5, 0.5, "No symbols demodulated", ha="center", va="center", color="#6B7684")

            # Decoded text box
            ax_text = fig.add_axes([0.52, 0.10, 0.40, 0.28], facecolor="#10151D")
            ax_text.set_title("Decoded Output Preview", fontsize=10, color="#E4E9F0", pad=6)
            dec_txt = self.lbl_decode.toPlainText()[:450]
            if not dec_txt:
                dec_txt = "No decoded payload."
            ax_text.text(0.04, 0.92, dec_txt, transform=ax_text.transAxes, fontsize=7.5,
                         fontfamily="monospace", color="#55EFC4", va="top", wrap=True)
            ax_text.axis("off")
            rect = plt.Rectangle((0,0), 1, 1, transform=ax_text.transAxes, fill=False, edgecolor="#1E2733", lw=1)
            ax_text.add_patch(rect)

            pdf.savefig(fig)
            plt.close(fig)

    # ------------------------------------------------------------------
    # Result handler
    # ------------------------------------------------------------------

    def _on_done(self, result: dict):
        info  = result["info"]
        spec  = result["spec"]
        cs    = result["constel"]
        freqs_w, times_w, Sdb_w = result["wfall"]
        fs    = result["fs"]

        self._current_samples = info["samples"]
        self._current_fs = fs
        self._current_spec = spec
        self._current_info = info
        self._pipeline_stage = "loaded"

        fname = os.path.basename(info["file_path"])
        is_wav = os.path.splitext(info["file_path"])[1].lower() == ".wav"

        # Rate estimation suggestion handler
        rate_est = result.get("rate_est")
        if rate_est and not is_wav:
            best_rate = rate_est["best_rate"]
            conf = rate_est["confidence"]
            self._suggested_fs = best_rate
            if best_rate == fs:
                self.lbl_rate_est.setText(f"Suggested: {best_rate/1e3:.0f} kHz ({conf:.0f}% match)")
                self.btn_apply_fs.setEnabled(False)
            else:
                self.lbl_rate_est.setText(f"Suggested: {best_rate/1e3:.0f} kHz ({conf:.0f}%)")
                self.btn_apply_fs.setEnabled(True)
                self.btn_apply_fs.setToolTip(f"Reload using suggested Fs = {best_rate/1e3:.0f} kHz")
        elif is_wav:
            self.lbl_rate_est.setText("(WAV embedded Fs)")
            self.btn_apply_fs.setEnabled(False)
        else:
            self.lbl_rate_est.setText("")
            self.btn_apply_fs.setEnabled(False)

        # Update Stepper
        fmt = info.get("iq_format") or f"{info.get('bit_depth','')}b wav"
        self.stage_load.set_state("done", f".iq · {fmt}")
        self.stage_classify.set_state("active", "Ready to classify")
        self._set_status_pill("● File Loaded", state="ready")

        # Top bar file field
        self.lbl_file.setText(
            f"Loaded — <b style='color:#E4E9F0;'>{fname}</b> &nbsp;|&nbsp; "
            f"<span style='color:#E4E9F0;'>{fs/1e6:.3f} MHz</span> fs &nbsp;|&nbsp; "
            f"<span style='color:#E4E9F0;'>{info['sample_count']:,}</span> samples"
        )

        # Footer Readout Bar
        self.lbl_readout_snr.setText(f"SNR  <b>{spec['snr_db']:.1f} dB</b>")
        self.lbl_readout_cfo.setText(f"CFO  <b>{spec['center_freq']:+.0f} Hz</b>")
        self.lbl_readout_phase.setText(f"Phase  <b>—</b>")
        self.lbl_readout_bw.setText(f"OccBW  <b>{spec['occupied_bw']/1e3:.1f} kHz</b>")
        self.lbl_readout_samples.setText(f"Samples  <b>{info['sample_count']:,}</b>")

        # Resume pipeline if triggered by Apply-Fs
        resume = getattr(self, "_resume_stage", None)
        self._resume_stage = None
        if resume in ("classified", "demodulated", "synced", "decoded"):
            self._classify()
        if resume in ("demodulated", "synced", "decoded") and getattr(self, "_last_predicted_mod", None):
            self._demodulate()
        if resume in ("synced", "decoded") and getattr(self, "_current_bits", None) is not None:
            self._find_sync()

        # Legacy lbl_info for compatibility
        dur = info.get("duration") or 0
        self.lbl_info.setText(
            f"Samples: {info['sample_count']:,}   Fs: {fs/1e3:.0f} kHz   "
            f"Duration: {dur:.2f}s   Format: {fmt}   "
            f"Centre: {spec['center_freq']:+.0f} Hz   "
            f"OccBW: {spec['occupied_bw']/1e3:.1f} kHz   "
            f"SNR~{spec['snr_db']:.0f} dB"
        )

        # --- Spectrum tab ---
        self.canvas_spec.fig.clear()
        ax = self.canvas_spec.fig.add_subplot(111)
        kHz = spec["freqs"] / 1e3
        ax.plot(kHz, spec["psd_db"], color="#4EE1B8", lw=1.0, label="PSD")
        ax.fill_between(kHz, spec["psd_db"], spec["psd_db"].min(),
                        alpha=0.10, color="#4EE1B8")
        ax.axhline(spec["peak_power_db"],   color="#F0A84E", ls="--", lw=1.1,
                   label=f"Peak {spec['peak_power_db']:.1f} dBFS")
        ax.axhline(spec["noise_floor_db"],  color="#414C59", ls=":",  lw=0.9,
                   label=f"Floor {spec['noise_floor_db']:.1f} dBFS")
        ax.axvline(spec["center_freq"]/1e3, color="#4EE1B8", ls="-.", lw=0.9, alpha=0.7,
                   label=f"Centre {spec['center_freq']:+.0f} Hz")
        ax.set_xlabel("Frequency (kHz)", color="#6B7684", fontsize=9)
        ax.set_ylabel("Power (dBFS)", color="#6B7684", fontsize=9)
        ax.set_title(f"Spectrum — {fname}", color="#E4E9F0", fontsize=10, pad=8)
        ax.legend(fontsize=8, facecolor="#10151D", edgecolor="#1E2733",
                  labelcolor="#E4E9F0", loc="upper right")
        ax.grid(True, lw=0.5, color="#1E2733", linestyle=":")
        self.canvas_spec.draw()

        # --- Waterfall tab ---
        self.canvas_wfall.fig.clear()
        ax2 = self.canvas_wfall.fig.add_subplot(111)
        vmax = Sdb_w.max()
        vmin = vmax - 60
        kHz_w = freqs_w / 1e3
        im = ax2.imshow(Sdb_w, origin="lower", aspect="auto",
                        extent=[times_w[0], times_w[-1], kHz_w[0], kHz_w[-1]],
                        cmap="inferno", vmin=vmin, vmax=vmax,
                        interpolation="nearest")
        cbar = self.canvas_wfall.fig.colorbar(im, ax=ax2, label="dBFS", fraction=0.03)
        cbar.ax.yaxis.label.set_color("#6B7684")
        cbar.ax.tick_params(colors="#6B7684", labelsize=8)
        ax2.set_xlabel("Time (s)", color="#6B7684", fontsize=9)
        ax2.set_ylabel("Frequency (kHz)", color="#6B7684", fontsize=9)
        ax2.set_title(f"Waterfall — {fname}", color="#E4E9F0", fontsize=10, pad=8)
        self.canvas_wfall.draw()

        # --- Constellation tab ---
        self._plot_constellation(cs, title=f"Constellation — {fname}")

        self.status.showMessage(
            f"Loaded {fname} | {info['sample_count']:,} samples | "
            f"SNR ~{spec['snr_db']:.0f} dB | OccBW {spec['occupied_bw']/1e3:.1f} kHz"
        )
        self._update_button_states(busy=False)

    def _plot_constellation(self, cs: np.ndarray, title: str | None = None):
        """Render I/Q scatter and density map on the constellation tab."""
        if cs is None or len(cs) == 0:
            return

        self.canvas_constel.fig.clear()
        ax3 = self.canvas_constel.fig.add_subplot(111)

        # Normalize to unit RMS power for consistent alignment to unit circle
        pwr = np.sqrt(np.mean(np.abs(cs) ** 2))
        cs_norm = cs / pwr if pwr > 0 else cs

        I = cs_norm.real
        Q = cs_norm.imag
        max_pts = 25_000
        if len(I) > max_pts:
            idx = np.random.choice(len(I), max_pts, replace=False)
            I, Q = I[idx], Q[idx]

        lim = 1.4
        theta = np.linspace(0, 2 * np.pi, 300)
        ax3.plot(np.cos(theta), np.sin(theta), color="#1E2733", lw=1.0, ls="--", zorder=1)
        ax3.axhline(0, color="#1E2733", lw=0.8, zorder=1)
        ax3.axvline(0, color="#1E2733", lw=0.8, zorder=1)

        hb = ax3.hexbin(
            I, Q, gridsize=60, cmap="inferno", mincnt=1, linewidths=0,
            extent=[-lim, lim, -lim, lim], zorder=2
        )
        cbar = self.canvas_constel.fig.colorbar(hb, ax=ax3, label="Density", fraction=0.046)
        cbar.ax.yaxis.label.set_color("#6B7684")
        cbar.ax.tick_params(colors="#6B7684", labelsize=8)
        ax3.scatter(I, Q, s=12, c="#4EE1B8", alpha=0.5, edgecolors="none", zorder=3)

        ax3.set_xlim(-lim, lim)
        ax3.set_ylim(-lim, lim)
        ax3.set_xlabel("In-Phase (I)", color="#6B7684", fontsize=9)
        ax3.set_ylabel("Quadrature (Q)", color="#6B7684", fontsize=9)
        ax3.grid(True, lw=0.5, color="#1E2733", linestyle=":")
        if title:
            ax3.set_title(title, color="#E4E9F0", fontsize=10, pad=8)
        ax3.set_aspect("equal")
        self.canvas_constel.draw()

    def _on_error(self, msg: str):
        self.btn_open.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.status.showMessage(f"ERROR: {msg}")
        self.lbl_info.setText(f"Error: {msg}")
        self.stage_load.set_state("pending", f"Error: {msg}")
        self._set_status_pill("● Error", state="error")
        self.show_banner(f"Failed to load signal file: {msg}", level="error")
        self._update_button_states(busy=False)

    # ------------------------------------------------------------------
    # Reset / Clear
    # ------------------------------------------------------------------

    def _reset_state(self):
        """Reset the app to its just-launched state without closing the window."""
        print("[Reset] _reset_state called — clearing all state...")
        try:
            if self._thread is not None and self._thread.isRunning():
                self._thread.quit()
                self._thread.wait()
        except RuntimeError:
            pass
        self._thread = None
        self._worker = None

        try:
            if self._pipeline_thread is not None and self._pipeline_thread.isRunning():
                self._pipeline_thread.quit()
                self._pipeline_thread.wait()
        except RuntimeError:
            pass
        self._pipeline_thread = None
        self._pipeline_worker = None

        self._current_file_path = None
        self._suggested_fs = None
        self._pipeline_stage = None
        self._resume_stage = None
        self._current_samples = None
        self._current_symbols = None
        self._current_bits = None
        self._current_payload_bits = None
        self._last_predicted_mod = None
        self._ground_truth_bits = None
        self._is_synthetic_file = False
        self._pipeline_low_confidence_flags = {}
        self._current_spec = None
        self._current_info = None

        self.hide_banner()
        self.viewport_stack.setCurrentIndex(0)  # Return to empty state placeholder

        self.lbl_file.setText("No file loaded")
        self.lbl_info.setText("—")
        self.lbl_class.setText("")
        self.lbl_demod.setPlainText("")
        self.lbl_sync.setPlainText("")
        self.lbl_decode.setPlainText("")
        self.txt_bitstream_view.setPlainText("")
        self.lbl_rate_est.setText("")

        self._set_status_pill("● Ready", state="ready")
        self.lbl_auto_fec_name.setText("Awaiting sync...")
        self.lbl_auto_fec_pct.setText("—")
        self.prog_auto_fec.setValue(0)
        self.lbl_auto_fec_runner1.setText("Runner-up: —")

        self.lbl_m_bits.setText("—")
        self.lbl_m_rate.setText("—")
        self.lbl_m_ber.setText("—")
        self.lbl_m_match.setText("—")

        self.lbl_readout_snr.setText("SNR  <b>—</b>")
        self.lbl_readout_cfo.setText("CFO  <b>—</b>")
        self.lbl_readout_phase.setText("Phase  <b>—</b>")
        self.lbl_readout_bw.setText("OccBW  <b>—</b>")
        self.lbl_readout_samples.setText("Samples  <b>—</b>")

        self.stage_load.set_state("pending", "No file loaded")
        self.stage_classify.set_state("pending", "Pending")
        self.stage_demod.set_state("pending", "Pending")
        self.stage_sync.set_state("pending", "Pending")
        self.stage_fec.set_state("pending", "Pending")
        self.stage_decode.set_state("pending", "Pending")

        self.btn_classify.setEnabled(False)
        self.btn_demod.setEnabled(False)
        self.btn_sync.setEnabled(False)
        self.btn_decode.setEnabled(False)
        self.btn_auto_fec.setEnabled(False)
        self.btn_apply_fs.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_report.setEnabled(False)
        self.cmb_fs.setEnabled(True)

        for canvas in (self.canvas_spec, self.canvas_wfall, self.canvas_constel):
            canvas.fig.clear()
            canvas.draw()

        self.status.showMessage("Ready — open an IQ or WAV file to begin.")
        print("[Reset] _reset_state called — all state cleared.")
