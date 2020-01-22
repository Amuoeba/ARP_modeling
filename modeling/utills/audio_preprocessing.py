# Imports from external libraries
import numpy as np
import struct
import webrtcvad
from scipy.ndimage.morphology import binary_dilation
from typing import Optional, Union
import librosa
from pathlib import Path
# Imports from internal libraries
import config


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * config.INT_16_MAX) ** 2))
    wave_dBFS = 20 * np.log10(rms / config.INT_16_MAX)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))

def moving_average(array, width):
    array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
    ret = np.cumsum(array_padded, dtype=float)
    ret[width:] = ret[width:] - ret[:-width]
    return ret[width - 1:] / width

def vad_trim_silences(wav):
    samples_per_window = (config.VAD_WIN_LEN * config.SR) // 1000
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * config.INT_16_MAX)).astype(np.int16))

    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=config.SR))

    voice_flags = np.array(voice_flags)
    audio_mask = moving_average(voice_flags, config.VAD_AVERAGE)
    audio_mask = np.round(audio_mask).astype(np.bool)

    audio_mask = binary_dilation(audio_mask, np.ones(config.VAD_MAX_SILENCE + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask == True]


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray], source_sr: Optional[int] = None):
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav
    if source_sr is not None:
        wav = librosa.resample(wav, source_sr, config.SR)

    wav = normalize_volume(wav, config.TAR_DBFS, increase_only=True)
    wav = vad_trim_silences(wav)

    return wav

def wav_to_mel_spectrogram(wav):
    frames = librosa.feature.melspectrogram(
        wav,
        config.SR,
        n_fft=int(config.SR * config.MEL_WIN_STEP / 1000),
        hop_length=int(config.SR * config.MEL_WIN_STEP / 1000),
        n_mels=config.MEL_CHANNELS
    )
    return frames.astype(np.float32).T
