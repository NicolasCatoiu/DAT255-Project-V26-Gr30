import io
import os
import sys
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR

SR                   = 16000
SILENCE_LABEL        = 10
CONFIDENCE_THRESHOLD = 0.70
MIC_STARTUP_SEC      = 0.30


def load_model():
    path = os.path.join(MODEL_DIR, "best_baselineModel_augmented.keras")
    return tf.keras.models.load_model(path, compile=False)


def _to_mono_16k(audio, sr):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio


def _normalize(audio):
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return audio / peak


def _rms(audio):
    return float(np.sqrt(np.mean(audio ** 2)))


def _trim_mic_startup(audio):
    check = int(SR * MIC_STARTUP_SEC)
    if len(audio) <= check * 2:
        return audio
    if _rms(audio[:check]) < 0.02:
        return audio[check:]
    return audio


def _pad_to_1s(audio):
    if len(audio) < SR:
        return np.pad(audio, (0, SR - len(audio)))
    return audio[:SR]


def _make_mel(audio_1s):
    mel = librosa.feature.melspectrogram(
        y=audio_1s, sr=SR, n_mels=64, n_fft=400, hop_length=160,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_norm[..., np.newaxis]


def predict_digit(audio_bytes, model):
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    audio = _to_mono_16k(audio, sr)
    audio = _normalize(audio)
    audio = _trim_mic_startup(audio)
    audio = _pad_to_1s(audio)
    mel   = _make_mel(audio)

    probs = model.predict(mel[np.newaxis], verbose=0)[0]
    label = int(np.argmax(probs))
    conf  = float(np.max(probs))

    if label != SILENCE_LABEL and conf >= CONFIDENCE_THRESHOLD:
        return {"digit": str(label), "confidence": conf, "heard": True}
    return {"digit": None, "confidence": conf, "heard": False}
