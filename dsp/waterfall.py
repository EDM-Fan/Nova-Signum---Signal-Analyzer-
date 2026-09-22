"""
dsp/waterfall.py
================
Spectrogram / waterfall display (frequency × time).

Main function
-------------
    plot_waterfall(samples, sample_rate, *, title, save_path, ...)

Also exports:
    compute_stft(samples, sample_rate, *, nfft, hop, window) -> (freqs, times, Sdb)
        Returns arrays suitable for imshow / external use.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# STFT computation
# ---------------------------------------------------------------------------

def compute_stft(
    samples: np.ndarray,
    sample_rate: int,
    *,
    nfft: int = 1024,
    hop: int | None = None,           # None -> nfft // 4
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Short-Time Fourier Transform of complex IQ.

    Returns
    -------
    freqs  : 1-D array of frequency values (Hz), centred at 0
    times  : 1-D array of time values (s)
    Sdb    : 2-D power array in dB, shape (nfft, n_frames)
             rows = frequency, columns = time
    """
    from scipy.signal import get_window

    samples = np.asarray(samples, dtype=np.complex128)
    n = len(samples)

    hop = hop if hop is not None else nfft // 4
    win = get_window(window, nfft)
    win_power = np.sum(win ** 2)

    n_frames = 1 + (n - nfft) // hop
    Sdb = np.empty((nfft, n_frames), dtype=np.float64)

    for k in range(n_frames):
        seg = samples[k * hop : k * hop + nfft] * win
        sp  = np.fft.fftshift(np.fft.fft(seg, n=nfft))
        Sdb[:, k] = 10 * np.log10(
            np.maximum(np.abs(sp) ** 2 / (win_power * sample_rate), 1e-30)
        )

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    times = np.arange(n_frames) * hop / sample_rate

    return freqs, times, Sdb


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_waterfall(
    samples: np.ndarray,
    sample_rate: int,
    *,
    title: str = "Waterfall (Spectrogram)",
    save_path: str | None = None,
    nfft: int = 1024,
    hop: int | None = None,
    window: str = "hann",
    colormap: str = "inferno",
    dynamic_range_db: float = 60.0,   # clip range below peak
    max_time_bins: int = 800,          # downsample time axis for display
) -> None:
    """
    Plot frequency vs time waterfall.

    Parameters
    ----------
    samples          : complex IQ array
    sample_rate      : Fs in Hz
    title            : plot title
    save_path        : save PNG here; if None, show interactively
    nfft             : FFT size per frame
    hop              : hop size in samples (default nfft//4)
    window           : window function name
    colormap         : matplotlib colormap for power
    dynamic_range_db : colour range below the peak (larger = more detail)
    max_time_bins    : max columns to display (decimates time axis if needed)
    """
    import matplotlib
    matplotlib.use("Agg" if save_path else "TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    freqs, times, Sdb = compute_stft(samples, sample_rate,
                                      nfft=nfft, hop=hop, window=window)

    # Downsample time axis for display performance
    n_frames = Sdb.shape[1]
    if n_frames > max_time_bins:
        step = n_frames // max_time_bins
        Sdb   = Sdb[:, ::step]
        times = times[::step]

    # Clip dynamic range
    vmax = Sdb.max()
    vmin = vmax - dynamic_range_db

    kHz = freqs / 1e3

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0d1117")

    im = ax.imshow(
        Sdb,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], kHz[0], kHz[-1]],
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Power (dBFS)", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("Time (s)", color="white", fontsize=11)
    ax.set_ylabel("Frequency (kHz)", color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(False)

    # Info text
    freq_res = sample_rate / nfft
    hop_used = hop if hop else nfft // 4
    time_res = hop_used / sample_rate * 1000  # ms
    info = (f"NFFT={nfft}  Freq res={freq_res:.0f} Hz  "
            f"Time res={time_res:.1f} ms  Window={window}")
    ax.text(0.01, 0.02, info, transform=ax.transAxes,
            fontsize=7, color="#aaa", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", ec="#555", alpha=0.7))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Waterfall saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)
