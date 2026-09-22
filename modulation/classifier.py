"""
Automatic modulation classification.
Extracts hand-crafted signal-processing features from I/Q samples and
classifies among BPSK / QPSK / 16QAM / 2FSK using a Random Forest.
"""
import os
import numpy as np
from scipy import signal as sps_signal
from scipy.stats import kurtosis
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    from modulation.signal_gen import generate_random_example, GENERATORS
except ImportError:
    from signal_gen import generate_random_example, GENERATORS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "classifier_model.joblib")
CLASSES = list(GENERATORS.keys())  # ['BPSK', 'QPSK', '16QAM', '2FSK']


def extract_features(samples, sample_rate):
    """
    Extract a feature vector from complex I/Q samples.
    Returns a 1D numpy array of features.
    """
    samples = np.asarray(samples)
    if len(samples) < 64:
        raise ValueError("Signal too short for feature extraction")

    amp = np.abs(samples)
    amp_norm = amp / (np.mean(amp) + 1e-12)

    # --- Amplitude / envelope statistics ---
    amp_var = np.var(amp_norm)
    amp_kurtosis = kurtosis(amp_norm, fisher=True)

    # --- Instantaneous phase & frequency ---
    inst_phase = np.unwrap(np.angle(samples))
    inst_freq = np.diff(inst_phase) * sample_rate / (2 * np.pi)
    inst_freq = inst_freq[np.isfinite(inst_freq)]
    freq_std = np.std(inst_freq) if len(inst_freq) else 0.0
    freq_kurtosis = kurtosis(inst_freq, fisher=True) if len(inst_freq) > 8 else 0.0

    # Phase of the "centered" nonlinear amplitude signal — captures PSK's
    # discrete phase jumps vs FSK's continuous phase ramp.
    phase_diff = np.diff(inst_phase)
    phase_diff_std = np.std(phase_diff)

    # --- Spectral features ---
    freqs, psd = sps_signal.welch(samples, fs=sample_rate, nperseg=min(1024, len(samples)), return_onesided=False)
    psd_norm = psd / (np.sum(psd) + 1e-12)
    spectral_flatness = np.exp(np.mean(np.log(psd_norm + 1e-12))) / (np.mean(psd_norm) + 1e-12)
    # count spectral peaks above half-max (BPSK/QPSK/QAM -> 1 lobe, FSK -> 2 lobes)
    half_max = psd.max() / 2
    peaks, _ = sps_signal.find_peaks(psd, height=half_max, distance=len(psd) // 20 + 1)
    num_spectral_peaks = len(peaks)

    # --- Higher-order moments on raw I/Q (constant modulus check) ---
    c20 = np.mean(samples ** 2)
    c40 = np.mean(np.abs(samples) ** 4) - 2 * np.mean(np.abs(samples) ** 2) ** 2 - np.abs(c20) ** 2
    c40_abs = np.abs(c40)

    # --- Constellation-based features (after crude symbol decimation) ---
    # Decimate to approximate symbol rate by taking every ~8th sample near
    # peak energy — a rough proxy since we don't have true timing recovery.
    decim = samples[::8]
    I = np.real(decim)
    Q = np.imag(decim)
    iq_ratio = np.std(I) / (np.std(Q) + 1e-12)
    # cluster spread via simple binning of angle (captures 2 vs 4 vs 16 clusters)
    angles = np.angle(decim)
    angle_hist, _ = np.histogram(angles, bins=16, range=(-np.pi, np.pi))
    angle_hist_norm = angle_hist / (np.sum(angle_hist) + 1e-12)
    angle_entropy = -np.sum(angle_hist_norm * np.log(angle_hist_norm + 1e-12))

    features = np.array([
        amp_var,
        amp_kurtosis,
        freq_std,
        freq_kurtosis,
        phase_diff_std,
        spectral_flatness,
        num_spectral_peaks,
        c40_abs,
        iq_ratio,
        angle_entropy,
    ])
    return features


FEATURE_NAMES = [
    "amp_var", "amp_kurtosis", "freq_std", "freq_kurtosis", "phase_diff_std",
    "spectral_flatness", "num_spectral_peaks", "c40_abs", "iq_ratio", "angle_entropy",
]


def build_training_set(n_examples=1200, sample_rate=1_000_000, num_symbols=2000, sps=8):
    X, y = [], []
    for i in range(n_examples):
        sig, mod, fs = generate_random_example(sample_rate, num_symbols, sps)
        try:
            feats = extract_features(sig, fs)
        except ValueError:
            continue
        X.append(feats)
        y.append(mod)
    return np.array(X), np.array(y)


def train_model(n_examples=1200, save=True, verbose=True):
    if verbose:
        print(f"Generating {n_examples} synthetic training examples...")
    X, y = build_training_set(n_examples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    if verbose:
        print(f"Held-out test accuracy: {acc*100:.1f}%  (n={len(y_test)})")
        importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
        print("Top features:")
        for name, imp in importances[:5]:
            print(f"  {name:<20} {imp:.3f}")

    if save:
        joblib.dump(clf, MODEL_PATH)
        if verbose:
            print(f"Model saved to {MODEL_PATH}")
    return clf, acc


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No trained model found at {MODEL_PATH}. Run train_model() first.")
    return joblib.load(MODEL_PATH)


def predict(samples, sample_rate, model=None):
    """
    Predict modulation type for a given complex IQ sample array.
    Returns (predicted_label: str, confidence_pct: float).
    """
    if model is None:
        model = load_model()
    feats = extract_features(samples, sample_rate).reshape(1, -1)
    probs = model.predict_proba(feats)[0]
    idx = np.argmax(probs)
    label = model.classes_[idx]
    confidence = probs[idx] * 100
    return label, confidence


if __name__ == "__main__":
    train_model()
