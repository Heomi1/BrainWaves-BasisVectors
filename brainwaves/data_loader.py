"""Load and preprocess resting-state EEG from the PhysioNet EEG Motor
Movement/Imagery dataset (EEGBCI) via MNE.

Runs used in this project:
    1 -- resting state, eyes open
    2 -- resting state, eyes closed (strong occipital alpha rhythm)
"""

import mne
from mne.datasets import eegbci

EYES_OPEN = 1
EYES_CLOSED = 2


def load_raw(subject=1, run=EYES_CLOSED, l_freq=1.0, h_freq=40.0):
    """Download (if needed) one EEGBCI recording and return a filtered
    ``mne.io.Raw`` object containing only the EEG channels.

    The 1-40 Hz band-pass removes DC drift and high-frequency noise.
    """
    edf_files = eegbci.load_data(subjects=[subject], runs=[run])
    raw = mne.io.read_raw_edf(edf_files[0], preload=True)
    eegbci.standardize(raw)  # "O1.." -> "O1", etc.
    raw.filter(l_freq, h_freq, fir_design="firwin")
    raw.pick("eeg")
    return raw


def load_data(subject=1, run=EYES_CLOSED):
    """Return ``(data, raw)`` where ``data`` has shape (channels, samples),
    e.g. (64, n_samples) for 64-channel EEG sampled at 160 Hz."""
    raw = load_raw(subject, run)
    return raw.get_data(), raw


if __name__ == "__main__":
    data, raw = load_data()
    print(f"Data matrix shape (channels, samples): {data.shape}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
