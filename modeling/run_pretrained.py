# Imports from external libraries
from pathlib import Path
import numpy as np
from tqdm import tqdm
from itertools import groupby

# Imports from internal libraries
from encoder.lstm_encoder import VoiceEncoder
import utills.audio_preprocessing as audioPrep
from utills.plotting import plot_projections
import config

if __name__ == '__main__':
    if config.PRETRAINED_MODE == "test":
        print("Running pretrained model")
        wav_fpaths = list(Path("audio_data", "librispeech_test-other").glob("**/*.flac"))
        wav_fpaths += list(Path("audio_data", "librispeech_test-other").glob("**/*.mp3"))
        wav_fpaths += list(Path("audio_data", "librispeech_test-other").glob("**/*.m4a"))
        speakers = list(map(lambda wav_fpath: wav_fpath.parent.stem, wav_fpaths))
        wavs = np.array(list(map(audioPrep.preprocess_wav, tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths)))))
        speaker_wavs = {speaker: wavs[list(indices)] for speaker, indices in
                        groupby(range(len(wavs)), lambda i: speakers[i])}

        encoder = VoiceEncoder(device="cpu")
        utterance_embeds = np.array(list(map(encoder.embed_utterance, wavs)))

        plot_projections(utterance_embeds, speakers, title="Embedding projections")


    elif config.PRETRAINED_MODE == "trace":
        print("Tracing the moddel")
        model = VoiceEncoder(device="cpu")
        model.export()