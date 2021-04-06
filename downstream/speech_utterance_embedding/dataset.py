from random import randint
from pathlib import Path
import pandas as pd

from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file

EFFECTS = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]


class SpeechCommandsBaseDataset(Dataset):
    """Tog job based dataset."""

    def __init__(self):
        self.data = []

    def __getitem__(self, idx):
        anchor_audio_path,pos_audio_path,neg_audio_path = self.data[idx]
        anchor_wav ,pos_wav ,neg_wav  = apply_effects_file(str(anchor_audio_path), EFFECTS),apply_effects_file(str(pos_audio_path), EFFECTS),apply_effects_file(str(neg_audio_path), EFFECTS)
        anchor_wav, pos_wav, neg_wav = anchor_wav[0].squeeze(0), pos_wav[0].squeeze(0), neg_wav[0].squeeze(0)
        return anchor_wav, pos_wav, neg_wav

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        """Collate a mini-batch of data."""
        anchor_wavs, pos_wavs, neg_wavs = zip(*samples)
        return  anchor_wavs, pos_wavs, neg_wavs


class SpeechCommandsDataset(SpeechCommandsBaseDataset):
    """Training and validation dataset."""

    def __init__(self, data_list, **kwargs):
        super().__init__()

        self.data = data_list

    def __getitem__(self, idx):
        anchor_wav, pos_wav, neg_wav = super().__getitem__(idx)
        
        return anchor_wav, pos_wav, neg_wav


