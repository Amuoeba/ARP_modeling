# Imports from external libraries
from pathlib import Path
from typing import Union, List
from torch import nn
from time import perf_counter as timer
import numpy as np
import torch
# Imports from internal libraries
import config
import utills.audio_preprocessing as audioPrep

class VoiceEncoder(nn.Module):
    def __init__(self, device: Union[str, torch.device] = None, verbose=True):
        super().__init__()

        self.lstm = nn.LSTM(config.MEL_CHANNELS, 256, 3, batch_first=True)
        self.linear = nn.Linear(256, 256)
        self.relu = nn.ReLU()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        weights_fpath = Path(__file__).resolve().parent.joinpath("pretrained.pt")

        checkpoint = torch.load(weights_fpath, map_location="cpu")
        self.load_state_dict(checkpoint["model_state"], strict=False)
        self.to(device)

    def forward(self, mels: torch.FloatTensor):
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))

        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    @staticmethod
    def compute_partial_slices(n_samples: int, rate, min_coverage):
        assert 0 < min_coverage <= 1

        samples_per_frame = int((config.SR * config.MEL_WIN_STEP / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = int(np.round((config.SR / rate) / samples_per_frame))

        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - 160 + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + 160])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices

    def embed_utterance(self, wav: np.ndarray, return_partials=False, rate=1.3, min_coverage=0.75):
        wav_slices, mel_slices = self.compute_partial_slices(len(wav), rate, min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        mel = audioPrep.wav_to_mel_spectrogram(wav)
        mels = np.array([mel[s] for s in mel_slices])
        with torch.no_grad():
            mels = torch.from_numpy(mels).to(self.device)
            partial_embeds = self(mels).cpu().numpy()

        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)

        if return_partials:
            return embed, partial_embeds, wav_slices
        return embed

    def export(self):
        self.eval()
        example = torch.rand(2, 160, 40)
        traced_script = torch.jit.trace(self, example)
        traced_script.save("./my_traced.pt")