import numpy as np
import librosa


def normalize_mfcc_batches(mfcc_batches):
    """
    Normalize MFCC batches using mean-variance normalization.

    Parameters:
    - mfcc_batches: List of MFCC arrays (each with shape (time_steps, n_mfcc)).

    Returns:
    - List of normalized MFCC arrays.
    """
    # Concatenate all batches along the time axis to compute global mean and std
    all_mfcc = np.vstack(mfcc_batches)  # Shape: (total_time_steps, n_mfcc)

    # Compute mean and standard deviation across all batches
    mean = np.mean(all_mfcc, axis=0)
    std = np.std(all_mfcc, axis=0)

    # Normalize each batch
    normalized_batches = [(batch - mean) / std for batch in mfcc_batches]

    return np.array(normalized_batches)


def extract_mfcc_with_sliding_window(
    file_path, batch_duration=5, hop_duration=5, n_mfcc=13, sr=16000
):
    """
    Extract MFCC features from audio in fixed-duration batches with a sliding window.

    Parameters:
    - file_path: Path to the audio file.
    - batch_duration: Duration of each batch (in seconds).
    - hop_duration: Hop duration between batches (in seconds).
    - n_mfcc: Number of MFCC coefficients to extract.
    - sr: Sampling rate for loading the audio.

    Returns:
    - List of MFCC arrays for each batch (shape: (time_steps, n_mfcc)).
    """

    # Load the audio file
    audio, sample_rate = librosa.load(file_path, sr=sr)

    # Calculate the number of samples for each batch and hop
    samples_per_batch = int(batch_duration * sample_rate)
    hop_length = int(hop_duration * sample_rate)

    mfcc_batches = []

    # Loop through audio using sliding window
    for start in range(0, len(audio) - samples_per_batch + 1, hop_length):
        # Extract the current batch
        batch = audio[start: start + samples_per_batch]

        # Compute MFCCs for the batch
        mfcc = librosa.feature.mfcc(y=batch, sr=sample_rate, n_mfcc=n_mfcc)

        # Transpose to shape (time_steps, n_mfcc)
        mfcc_batches.append(mfcc.T)

    return mfcc_batches

def get_mfcc_features(file):
    mfcc_batches = extract_mfcc_with_sliding_window(file)
    # You can perform any other operation here like saving the MFCCs, further processing, etc.
    normalized_mfcc_batches = normalize_mfcc_batches(mfcc_batches)
    return normalized_mfcc_batches


