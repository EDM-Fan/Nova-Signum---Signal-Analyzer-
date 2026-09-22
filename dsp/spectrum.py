"""
dsp/spectrum.py
===============
FFT / Power Spectral Density analysis for complex IQ signals.

Main function
-------------
    analyse(samples, sample_rate, **kwargs) -> dict

Returns
-------
dict with keys:
    freqs         : np.ndarray  frequency axis (Hz), centred at 0
    psd_db        : np.ndarray  power spectral density (dBFS/Hz approx.)
    freq_resolution: float      Hz per FFT bin
    center_freq   : float       estimated centre of occupied spectrum (Hz)
    occupied_bw   : float       occupied bandwidth at -3 dB from peak (Hz)
    peak_power_db : float       peak PSD value (dBFS)
    noise_floor_db: float       estimated noise floor (dBFS)
    snr_db        : float       estimated SNR (dB)
    nfft          : int         FFT size actually used
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def _make_window(name: str, nfft: int) -> np.ndarray:
    windows = {
        "hann":        np.hanning,
        "hamming":     np.hamming,
        "blackman":    np.blackman,
        "bartlett":    np.bartlett,
        "rectangular": np.ones,
    }
    fn = windows.get(name.lower(), np.hanning)
    return fn(nfft)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse(
    samples: np.ndarray,
    sample_rate: int,
    *,
    nfft: int | None = None,
    window: str = "hann",
    overlap: float = 0.5,
    average: str = "mean",          # "mean" or "max"
) -> dict:
    """
    Compute averaged PSD via Welch-style segmented FFT.

    Parameters
    ----------
    samples     : complex IQ array
    sample_rate : samples per second
    nfft        : FFT size (default: min(8192, len(samples)))
    window      : window function name
    overlap     : segment overlap fraction [0, 1)
    average     : how to combine segment PSDs ('mean' or 'max')
    """
    samples = np.asarray(samples, dtype=np.complex128)
    n = len(samples)

    if nfft is None:
        nfft = min(8192, n)

    hop  = max(1, int(nfft * (1 - overlap)))
    win  = _make_window(window, nfft)
    win2 = np.sum(win ** 2)            # power normalisation factor

    # Segment the signal
    n_segs = max(1, 1 + (n - nfft) // hop)
    psd_accum = None

    for k in range(n_segs):
        seg = samples[k * hop : k * hop + nfft]
        if len(seg) < nfft:
            seg = np.pad(seg, (0, nfft - len(seg)))
        seg_w = seg * win
        sp    = np.fft.fft(seg_w, n=nfft)
        sp    = np.fft.fftshift(sp)
        power = (np.abs(sp) ** 2) / (win2 * sample_rate)

        if psd_accum is None:
            psd_accum = power
        else:
            if average == "max":
                psd_accum = np.maximum(psd_accum, power)
            else:
                psd_accum += power

    if average == "mean":
        psd_accum /= n_segs

    psd_accum = np.maximum(psd_accum, 1e-30)   # guard against log(0)
    psd_db    = 10 * np.log10(psd_accum)

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    freq_res = sample_rate / nfft

    # --- derived metrics ---
    peak_idx      = int(np.argmax(psd_db))
    peak_power_db = float(psd_db[peak_idx])

    # Noise floor: median of lower 40 % of PSD values
    sorted_psd    = np.sort(psd_db)
    noise_floor_db = float(np.median(sorted_psd[: int(0.4 * nfft)]))
    snr_db         = peak_power_db - noise_floor_db

    # Centre frequency: power-weighted centroid of bins above noise+3 dB
    threshold = noise_floor_db + 3.0
    mask      = psd_db >= threshold
    if mask.sum() == 0:
        mask = np.ones(nfft, dtype=bool)
    weights      = np.where(mask, 10 ** (psd_db / 10.0), 0.0)
    center_freq  = float(np.sum(freqs * weights) / np.sum(weights))

    # Occupied BW: span of bins whose power > (peak - 3 dB)
    bw_mask     = psd_db >= (peak_power_db - 3.0)
    occ_freqs   = freqs[bw_mask]
    occupied_bw = float(occ_freqs[-1] - occ_freqs[0]) if len(occ_freqs) > 1 else freq_res

    return {
        "freqs":          freqs,
        "psd_db":         psd_db,
        "freq_resolution":freq_res,
        "center_freq":    center_freq,
        "occupied_bw":    occupied_bw,
        "peak_power_db":  peak_power_db,
        "noise_floor_db": noise_floor_db,
        "snr_db":         snr_db,
        "nfft":           nfft,
        "n_segments":     n_segs,
    }


def print_metrics(result: dict) -> None:
    """Print the key spectral metrics from analyse()."""
    print(f"  FFT size        : {result['nfft']}  ({result['n_segments']} segments averaged)")
    print(f"  Freq resolution : {result['freq_resolution']:.2f} Hz/bin")
    print(f"  Centre freq     : {result['center_freq']:+.1f} Hz")
    print(f"  Occupied BW     : {result['occupied_bw']/1e3:.2f} kHz")
    print(f"  Peak power      : {result['peak_power_db']:.1f} dBFS")
    print(f"  Noise floor     : {result['noise_floor_db']:.1f} dBFS")
    print(f"  Est. SNR        : {result['snr_db']:.1f} dB")


def plot_spectrum(
    result: dict,
    title: str = "Power Spectral Density",
    save_path: str | None = None,
) -> None:
    """
    Plot the PSD with annotated metrics.

    Parameters
    ----------
    result    : dict from analyse()
    title     : plot title
    save_path : if given, save figure to this path instead of showing
    """
    import matplotlib
    matplotlib.use("Agg" if save_path else "TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    freqs  = result["freqs"]
    psd_db = result["psd_db"]
    kHz    = freqs / 1e3

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    ax.plot(kHz, psd_db, color="#00d4ff", linewidth=0.8, label="PSD")
    ax.fill_between(kHz, psd_db, psd_db.min(), alpha=0.15, color="#00d4ff")

    # Annotate lines
    ax.axhline(result["peak_power_db"],   color="#ff6b6b", ls="--", lw=1.2,
               label=f"Peak {result['peak_power_db']:.1f} dBFS")
    ax.axhline(result["noise_floor_db"],  color="#ffd93d", ls=":",  lw=1.0,
               label=f"Noise floor {result['noise_floor_db']:.1f} dBFS")
    ax.axvline(result["center_freq"]/1e3, color="#6bcb77", ls="-.", lw=1.2,
               label=f"Centre {result['center_freq']:+.0f} Hz")

    # Shade occupied bandwidth
    f_low  = (result["center_freq"] - result["occupied_bw"] / 2) / 1e3
    f_high = (result["center_freq"] + result["occupied_bw"] / 2) / 1e3
    ax.axvspan(f_low, f_high, alpha=0.10, color="#6bcb77",
               label=f"Occ. BW {result['occupied_bw']/1e3:.1f} kHz")

    ax.set_xlabel("Frequency (kHz)", color="white", fontsize=11)
    ax.set_ylabel("Power (dBFS)", color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(True, color="#333", linewidth=0.5)
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white",
              fontsize=9, loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Spectrum plot saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)
