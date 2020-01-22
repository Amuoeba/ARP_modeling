# Imports from external libraries
from torch.utils.data import Dataset, DataLoader
import random
import webrtcvad
import librosa
import numpy as np
import struct
from scipy.ndimage.morphology import binary_dilation
from typing import Type
# Imports from internal libraries
import config
import database as db
from utills.plotting import plot_MFCC


class SpectrogramPairsDataset(Dataset):
    int16_max = (2 ** 15) - 1
    samplingRate = 16000
    audio_norm_target_dBFS = -30
    vad_window_length = 30
    vad_moving_average_width = 8
    vad_max_silence_length = 6

    mel_window_length = 25
    mel_window_step = 10
    mel_n_channels = 40

    def __init__(self, database, mode):
        self.database = db.FileDatabase(config.DATABASE).connect()
        self.mode = mode
        if self.mode == "train":
            self.table = db.TABLE_train
        elif self.mode == "test":
            self.table = db.TABLE_test
        else:
            raise NameError(f"Wrong mode: {self.mode}. Possible 'train' or 'test'")
    @staticmethod
    def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
        if increase_only and decrease_only:
            raise ValueError("Both increase only and decrease only are set")
        rms = np.sqrt(np.mean((wav * SpectrogramPairsDataset.int16_max) ** 2))
        wave_dBFS = 20 * np.log10(rms / SpectrogramPairsDataset.int16_max)
        dBFS_change = target_dBFS - wave_dBFS
        if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
            return wav
        return wav * (10 ** (dBFS_change / 20))

    @staticmethod
    def trim_long_silences(wav):
        samples_per_window = (SpectrogramPairsDataset.vad_window_length * SpectrogramPairsDataset.samplingRate) // 1000

        wav = wav[:len(wav) - (len(wav) % samples_per_window)]

        pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * SpectrogramPairsDataset.int16_max)).astype(np.int16))

        voice_flags = []
        vad = webrtcvad.Vad(mode=3)
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                             sample_rate=SpectrogramPairsDataset.samplingRate))
        voice_flags = np.array(voice_flags)

        def moving_average(array, width):
            array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
            ret = np.cumsum(array_padded, dtype=float)
            ret[width:] = ret[width:] - ret[:-width]
            return ret[width - 1:] / width

        audio_mask = moving_average(voice_flags, SpectrogramPairsDataset.vad_moving_average_width)
        audio_mask = np.round(audio_mask).astype(np.bool)

        audio_mask = binary_dilation(audio_mask, np.ones(SpectrogramPairsDataset.vad_max_silence_length + 1))
        audio_mask = np.repeat(audio_mask, samples_per_window)

        return wav[audio_mask == True]

    @staticmethod
    def wav_to_mel_spectrogram(wav):
        sampling_rate = SpectrogramPairsDataset.samplingRate
        mel_window_lenght = SpectrogramPairsDataset.mel_window_length
        mel_window_steps = SpectrogramPairsDataset.mel_window_step
        mel_n_channels = SpectrogramPairsDataset.mel_n_channels

        frames = librosa.feature.melspectrogram(
            wav,
            sampling_rate,
            n_fft=int(sampling_rate * mel_window_lenght / 1000),  # 512,#
            hop_length=int(sampling_rate * mel_window_steps / 1000),  # 128,#
            n_mels=mel_n_channels
        )
        return frames.astype(np.float32).T


    def __len__(self):
        no_elements = self.database.get_num_elements(self.table)
        return no_elements

    def __getitem__(self, item):
        recording = self.database.get_nth_item(item, self.table)
        print(recording)
        idx = recording[0]
        id = recording[1]
        file_name = recording[4]
        pos_sample = self.database.get_random_same(self.table,idx,id)
        neg_sample = self.database.get_random_different(self.table,id)
        print(pos_sample)
        print(neg_sample)

        wav, source_sr = librosa.load(file_name, sr=None)
        if source_sr is not None:
            wav = librosa.resample(wav, source_sr, SpectrogramPairsDataset.samplingRate)
        else:
            print("Unknown source sampling rate")

        wav = self.normalize_volume(wav, SpectrogramPairsDataset.audio_norm_target_dBFS, increase_only=True)
        print(f"Recording time: {wav.shape[0]/16000}s")
        print("Before:",wav.shape)
        wav = self.trim_long_silences(wav)
        print("After:",wav.shape)
        mel_spectrogram  = self.wav_to_mel_spectrogram(wav)
        print("After mel:")

        print(mel_spectrogram.shape)
        print(type(mel_spectrogram))
        return mel_spectrogram,id,wav.shape[0]


if __name__ == '__main__':
    print("Testing dataset functionality")
    # torch.jit.trace()

    test_dataset = SpectrogramPairsDataset(config.DATABASE, "test")
    # print(test_dataset._get_num_files_())
    print(len(test_dataset))
    # print(test_dataset[1].shape)
    for i in [random.randrange(1, len(test_dataset), 1) for _ in range(10)]:
        mfcc,id,wav_samples = test_dataset[i]
        print(f"Id: {id} mfcc shape: {mfcc.shape} Image name: {config.IMAGES+id}")
        time = wav_samples/test_dataset.samplingRate
        plot_MFCC(mfcc,time,config.IMAGES+id)

