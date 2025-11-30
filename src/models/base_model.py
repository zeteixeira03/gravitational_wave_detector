import numpy as np
from scipy.signal import spectrogram
# STRATEGY HERE: 
# get a collection of aggregated features from every times series (1 per detector)
# use a basic logistic regression model to predict from those features

FS = 2048                          # sampling rate (Hz)
N = 4096                           # samples per detector
T = np.arange(N) / FS              # time axis (s)
DETECTORS = ["Hanford (H1)", "Livingston (L1)", "Virgo (V1)"]

def _compute_fft_band_energies(x: np.ndarray, fs: float, bands):
    """
    x: 1D array, time series of a single detector (length N)
    fs: sampling rate
    bands: list of (f_low, f_high) tuples in Hz

    returns: 1D array with band energies (one per band)
    """
    # One-sided FFT
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)     # shape (N/2 + 1,)
    fft_vals = np.fft.rfft(x)                       # complex spectrum
    power = np.abs(fft_vals) ** 2                   # power spectrum

    energies = []
    for f_low, f_high in bands:
        mask = (freqs >= f_low) & (freqs < f_high)
        # Sum power in this band (add small epsilon to avoid log(0))
        band_power = power[mask].sum()
        # log1p for numerical stability / dynamic range compression
        energies.append(np.log1p(band_power))

    return np.array(energies, dtype=np.float32)


def compute_features(sample: np.ndarray, fs: float = 2048.0) -> np.ndarray:
    """
    sample: np.ndarray of shape (3, 4096)  # H1, L1, V1

    returns: 1D np.ndarray of engineered features combining:
      - time-domain stats per detector
      - cross-detector correlations
      - FFT band energies per detector
    """
    feats = []

    # Define frequency bands (in Hz) – tweak as you like
    # Here we focus on the ~20–512 Hz range where most GW signal power lives
    bands = [
        (20, 60),
        (60, 120),
        (120, 250),
        (250, 400),
        (400, 512),
    ]

    # ----- Per-detector time-domain stats + band energies -----
    for det in range(sample.shape[0]):
        x = sample[det]

        # Time-domain stats
        mean = x.mean()
        std = x.std()
        max_abs = np.max(np.abs(x))
        rms = np.sqrt(np.mean(x ** 2))

        feats.extend([mean, std, max_abs, rms])

        # Frequency-domain features: band energies
        band_energies = _compute_fft_band_energies(x, fs=fs, bands=bands)
        feats.extend(band_energies.tolist())

    # ----- Cross-detector correlations (time domain) -----
    h, l, v = sample
    # Add small checks to avoid NaNs if variance is 0
    def safe_corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    feats.append(safe_corr(h, l))
    feats.append(safe_corr(h, v))
    feats.append(safe_corr(l, v))

    return np.array(feats, dtype=np.float32)
