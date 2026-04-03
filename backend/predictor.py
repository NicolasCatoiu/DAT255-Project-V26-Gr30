import io
import os
import sys
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR

DIGIT_LABELS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# Map "0"-"9" strings to model label words
INT_TO_WORD = {str(i): DIGIT_LABELS[i] for i in range(10)}
WORD_TO_INT = {v: str(i) for i, v in enumerate(DIGIT_LABELS)}

def load_model():
    path = os.path.join(MODEL_DIR, "deep_model-melSpectrogram.keras")
    return tf.keras.models.load_model(path, compile=False)


def _preprocess_segment(audio_array, sr=16000):
    """Convert a 1D float32 audio array to a model-ready mel spectrogram."""
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

    target_length = 16000
    if len(audio_array) < target_length:
        audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))
    else:
        audio_array = audio_array[:target_length]

    mel = librosa.feature.melspectrogram(
        y=audio_array, sr=16000, n_mels=64, n_fft=400, hop_length=160
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_norm[..., np.newaxis]  # (64, 101, 1)


def _segment_audio(audio_array, expected_count):
    """
    Split audio into individual digit segments using silence detection.
    Falls back to equal-length chunks if the wrong number of segments is found.
    """
    intervals = librosa.effects.split(audio_array, top_db=25, frame_length=512, hop_length=128)

    # Filter out very short noise bursts (< 0.08 s at 16 kHz)
    segments = [audio_array[s:e] for s, e in intervals if (e - s) > 1280]

    if len(segments) == expected_count:
        return segments

    # Fallback: equal-length chunks
    chunk = len(audio_array) // expected_count
    return [audio_array[i * chunk: (i + 1) * chunk] for i in range(expected_count)]


def predict_sequence(audio_bytes, expected_length, model=None):
    """
    Predict a spoken digit sequence from raw audio bytes.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio data (wav/webm/ogg — anything soundfile can read).
    expected_length : int
        Number of digits in the sequence (used for segmentation fallback).

    Returns
    -------
    list[str]
        Predicted digits as strings, e.g. ['3', '7', '1'].
    """
    audio_array, sr = sf.read(io.BytesIO(audio_bytes))

    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    audio_array = audio_array.astype(np.float32)

    segments = _segment_audio(audio_array, expected_length)

    if model is None:
        model = load_model()
    batch = np.stack([_preprocess_segment(seg, sr) for seg in segments])  # (N, 64, 101, 1)
    preds = model.predict(batch, verbose=0).argmax(axis=1)

    return [str(p) for p in preds]  # return as "0"-"9" strings to match game sequence
