import numpy as np

def time_shift(audio, max_shift_ms=50, sr=16000):
    """Shift audio left or right by a random amount"""
    max_shift_samples = int(sr * max_shift_ms / 1000)
    shift = np.random.randint(-max_shift_samples, max_shift_samples)
    return np.roll(audio, shift)


def volume_perturbation(audio, gain_range=(0.7, 1.3)):
    """Randomly scale volume to simulate mic distance variation"""
    gain = np.random.uniform(*gain_range)
    return np.clip(audio * gain, -1.0, 1.0)


def sample_silence_window(long_clip, target_length=16000):
    """Extract a random 1-second window from a longer silence clip"""
    if len(long_clip) <= target_length:
        return np.pad(long_clip, (0, target_length - len(long_clip)))
    max_offset = len(long_clip) - target_length
    offset = np.random.randint(0, max_offset)
    return long_clip[offset:offset + target_length]


def add_background_noise(audio, noise_clips, snr_db_range=(10, 25)):
    """Mix in background noise at random SNR"""
    noise_clip = noise_clips[np.random.randint(len(noise_clips))]
    noise = sample_silence_window(noise_clip)
    
    signal_power = np.mean(audio ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    
    snr_db = np.random.uniform(*snr_db_range)
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = np.sqrt(signal_power / (noise_power * snr_linear))
    
    return np.clip(audio + noise * noise_scale, -1.0, 1.0)


def augment_audio(audio, noise_clips=None, p=0.5):
    """Apply random combination of raw-audio augmentations"""
    if np.random.random() < p:
        audio = time_shift(audio)
    if np.random.random() < p:
        audio = volume_perturbation(audio)
    if noise_clips is not None and np.random.random() < p:
        audio = add_background_noise(audio, noise_clips)
    return audio


def spec_augment(spectrogram, n_freq_masks=1, n_time_masks=1,
                 freq_mask_param=8, time_mask_param=10):
    """SpecAugment — mask frequency and time bands in spectrogram"""
    spec = spectrogram.copy()
    freq_bins, time_steps, _ = spec.shape
    
    for _ in range(n_freq_masks):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, freq_bins - f)
        spec[f0:f0+f, :, :] = 0
    
    for _ in range(n_time_masks):
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, time_steps - t)
        spec[:, t0:t0+t, :] = 0
    
    return spec