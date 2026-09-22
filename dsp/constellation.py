"""
dsp/constellation.py
====================
I/Q scatter (constellation) diagram.

Main function
-------------
    plot_constellation(samples, sample_rate, *, title, save_path, max_points, ...)

Also exports:
    prepare_samples(samples, sample_rate, *, decimate_to, lp_cutoff) -> np.ndarray
        Light decimation + optional lowpass for cleaner scatter.
    compute_stats(samples) -> dict
        Cluster count estimate, RMS power, phase noise estimate.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Sample preparation
# ---------------------------------------------------------------------------

def prepare_samples(
    samples: np.ndarray,
    sample_rate: int,
    *,
    decimate_to: int = 100_000,   # target sample rate after decimation
    lp_cutoff: float = 0.45,       # normalised lowpass cutoff (×Nyquist)
) -> np.ndarray:
    """
    Optionally lowpass-filter and decimate complex IQ for cleaner constellation.

    Parameters
    ----------
    samples      : complex IQ array
    sample_rate  : original Fs (Hz)
    decimate_to  : target Fs after decimation (Hz); no decimation if >= sample_rate
    lp_cutoff    : FIR lowpass cutoff as fraction of Nyquist (0 < lp_cutoff < 1)
    """
    from scipy.signal import firwin, lfilter, decimate

    samples = np.asarray(samples, dtype=np.complex128)
    factor = max(1, sample_rate // decimate_to)

    if factor > 1:
        # Linear-phase FIR lowpass before decimation (prevents aliasing)
        taps = firwin(64, lp_cutoff / factor, window="hamming")
        # Filter I and Q separately (real coefficients → safe for complex)
        filt = (lfilter(taps, [1.0], samples.real)
                + 1j * lfilter(taps, [1.0], samples.imag))
        samples = filt[::factor]

    return samples


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------

def compute_stats(samples: np.ndarray) -> dict:
    """
    Rough stats useful for the GUI info panel.

    Returns
    -------
    dict with: rms_power_db, phase_std_deg, i_std, q_std
    """
    pwr = np.mean(np.abs(samples) ** 2)
    rms_db = 10 * np.log10(max(pwr, 1e-30))

    phases = np.angle(samples, deg=True)
    phase_std = float(np.std(phases))

    return {
        "rms_power_db": float(rms_db),
        "phase_std_deg": phase_std,
        "i_std": float(np.std(samples.real)),
        "q_std": float(np.std(samples.imag)),
        "n_samples": len(samples),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_constellation(
    samples: np.ndarray,
    sample_rate: int,
    *,
    title: str = "Constellation",
    save_path: str | None = None,
    max_points: int = 20_000,
    decimate: bool = True,
    alpha: float = 0.25,
    point_size: float = 2.0,
) -> None:
    """
    Plot an I/Q scatter diagram.

    Parameters
    ----------
    samples     : complex IQ array
    sample_rate : Fs in Hz
    title       : plot title
    save_path   : save PNG here; if None, display interactively
    max_points  : subsample to at most this many scatter points
    decimate    : whether to lowpass+decimate before plotting
    alpha       : scatter point transparency
    point_size  : scatter marker size
    """
    import matplotlib
    matplotlib.use("Agg" if save_path else "TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if decimate:
        plot_samp = prepare_samples(samples, sample_rate)
    else:
        plot_samp = np.asarray(samples, dtype=np.complex128)

    # Sub-sample for speed
    if len(plot_samp) > max_points:
        idx = np.random.choice(len(plot_samp), max_points, replace=False)
        idx.sort()
        plot_samp = plot_samp[idx]

    stats = compute_stats(plot_samp)

    I = plot_samp.real
    Q = plot_samp.imag

    # --- figure ---
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0d1117")

    # Density-coloured scatter via 2-D histogram rendered as image
    lim = max(np.abs(I).max(), np.abs(Q).max()) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Hex-bin density map
    hb = ax.hexbin(I, Q, gridsize=120, cmap="inferno",
                   mincnt=1, linewidths=0)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Point density", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # Reference circle (unit circle)
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta),
            color="#444", lw=0.8, ls="--", zorder=1)

    # Cross-hair
    ax.axhline(0, color="#333", lw=0.6)
    ax.axvline(0, color="#333", lw=0.6)

    ax.set_xlabel("In-Phase (I)", color="white", fontsize=11)
    ax.set_ylabel("Quadrature (Q)", color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # Stats box
    stats_text = (
        f"RMS power : {stats['rms_power_db']:.1f} dBFS\n"
        f"Phase std : {stats['phase_std_deg']:.1f} deg\n"
        f"Points    : {len(plot_samp):,}"
    )
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
            fontsize=8, color="white", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2e", ec="#555", alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Constellation saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)
