import io
import os
import sys
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR

SR = 16000
SILENCE_LABEL        = 10
CONFIDENCE_THRESHOLD = 0.60

TOP_DB           = 20     # speech detection sensitivity
MERGE_GAP_SEC    = 0.12   # merge intervals closer than this (same word)
MIN_WORD_SEC     = 0.10   # absolute minimum speech length
AMBIGUOUS_SEC    = 0.60   # segments 0.6–1.2s: try both interpretations
LONG_SEC         = 1.20   # segments > 1.2s: definitely multiple words
ENERGY_FLOOR_RMS = 3e-5   # skip segments quieter than this
MIC_STARTUP_SEC  = 0.30   # trim leading near-silence from browser mic

DIGIT_LABELS = [
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine", "silence",
]

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

def _trim_mic_startup(audio):
    check = int(SR * MIC_STARTUP_SEC)
    if len(audio) <= check * 2:
        return audio
    if _rms(audio[:check]) < 0.02:
        return audio[check:]
    return audio

def _rms(audio):
    return float(np.sqrt(np.mean(audio ** 2)))

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


def _find_energy_dip(audio):
    frame_len = int(SR * 0.025)
    hop       = int(SR * 0.010)
    if len(audio) < frame_len * 3:
        return None

    n_frames = (len(audio) - frame_len) // hop + 1
    energy = np.array([
        np.sum(audio[i * hop: i * hop + frame_len] ** 2)
        for i in range(n_frames)
    ])
    if len(energy) < 5:
        return None

    energy_smooth = np.convolve(energy, np.ones(5) / 5, mode="same")
    margin = len(energy_smooth) // 5
    region = energy_smooth[margin: -margin] if margin > 0 else energy_smooth
    if len(region) == 0:
        return None

    min_idx     = int(np.argmin(region)) + margin
    min_energy  = energy_smooth[min_idx]
    mean_energy = np.mean(energy_smooth)

    if min_energy > mean_energy * 0.4:
        return None

    return min_idx * hop


def _split_segment(audio):
    min_samples = int(SR * MIN_WORD_SEC)
    dip = _find_energy_dip(audio)
    if dip is None:
        return [audio]
    parts = [audio[:dip], audio[dip:]]
    return [p for p in parts if len(p) >= min_samples and _rms(p) >= ENERGY_FLOOR_RMS] or [audio]


def _find_words(audio):
    min_samples       = int(SR * MIN_WORD_SEC)
    min_gap           = int(SR * MERGE_GAP_SEC)
    long_samples      = int(SR * LONG_SEC)

    intervals = librosa.effects.split(audio, top_db=TOP_DB, frame_length=1024, hop_length=256)
    if len(intervals) == 0:
        return [], []

    # merge very close intervals (< 0.12s = same word)
    merged = [(int(intervals[0][0]), int(intervals[0][1]))]
    for start, end in intervals[1:]:
        start, end = int(start), int(end)
        if start - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    definite_words  = []
    ambiguous_words = []

    for start, end in merged:
        seg = audio[start:end]
        if len(seg) < min_samples or _rms(seg) < ENERGY_FLOOR_RMS:
            continue

        seg_sec = len(seg) / SR

        if seg_sec < AMBIGUOUS_SEC:
            # short — definitely one word
            definite_words.append(("single", _pad_to_1s(seg)))

        elif seg_sec < LONG_SEC:
            # ambiguous — try both whole and split
            whole = _pad_to_1s(seg)
            parts = _split_segment(seg)
            if len(parts) == 2:
                ambiguous_words.append({
                    "position": len(definite_words) + len(ambiguous_words),
                    "whole":    whole,
                    "parts":    [_pad_to_1s(p) for p in parts],
                })
            else:
                definite_words.append(("single", whole))

        else:
            # long — definitely multiple words, split recursively
            for p in _split_segment(seg):
                if len(p) >= min_samples:
                    if len(p) >= long_samples:
                        for s in _split_segment(p):
                            if len(s) >= min_samples:
                                definite_words.append(("single", _pad_to_1s(s)))
                    else:
                        definite_words.append(("single", _pad_to_1s(p)))

    return definite_words, ambiguous_words


def _classify_batch(mels, model):
    if not mels:
        return []
    probs = model.predict(np.stack(mels), verbose=0)
    return [(int(np.argmax(p)), float(np.max(p))) for p in probs]


def _is_valid_digit(label, conf):
    return label != SILENCE_LABEL and conf >= CONFIDENCE_THRESHOLD


def _predict_with_model(definite_words, ambiguous_words, model):
    results = []

    if definite_words:
        mels  = [_make_mel(w) for _, w in definite_words]
        for label, conf in _classify_batch(mels, model):
            if _is_valid_digit(label, conf):
                results.append({"digit": str(label), "confidence": conf})

    for amb in ambiguous_words:
        whole_pred = _classify_batch([_make_mel(amb["whole"])], model)[0]
        part_preds = _classify_batch([_make_mel(p) for p in amb["parts"]], model)
        parts_valid = [_is_valid_digit(*p) for p in part_preds]

        if all(parts_valid):
            for label, conf in part_preds:
                results.append({"digit": str(label), "confidence": conf})
        elif _is_valid_digit(*whole_pred):
            results.append({"digit": str(whole_pred[0]), "confidence": whole_pred[1]})
        elif any(parts_valid):
            for (label, conf), valid in zip(part_preds, parts_valid):
                if valid:
                    results.append({"digit": str(label), "confidence": conf})

    return results


def predict_sequence(audio_bytes, expected_length, model=None):
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    audio = _to_mono_16k(audio, sr)
    audio = _normalize(audio)
    audio = _trim_mic_startup(audio)

    if model is None:
        model = load_model()

    definite, ambiguous = _find_words(audio)
    all_digits = _predict_with_model(definite, ambiguous, model)
    collected  = all_digits[:expected_length]

    return {
        "sequence":     [d["digit"] for d in collected],
        "predictions":  collected,
        "heard_enough": len(collected) == expected_length,
    }
