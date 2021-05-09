from random import randint
from pathlib import Path
import pandas as pd

from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file

EFFECTS = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]


class SpeechCommandsBaseDataset(Dataset):
    """Tog job based dataset."""

    def __init__(self, CLASSES):
        self.class2index = {CLASSES[i]: i for i in range(len(CLASSES))}
        self.class_num = len(CLASSES)
        self.data = []

    def __getitem__(self, idx):
        audio_path, class_name = self.data[idx]
        wav, _ = apply_effects_file(str(audio_path), EFFECTS)
        wav = wav.squeeze(0)
        return wav, self.class2index[class_name], audio_path

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        """Collate a mini-batch of data."""
        wavs, labels, audio_paths = zip(*samples)
        return wavs, labels, audio_paths


class SpeechCommandsDataset(SpeechCommandsBaseDataset):
    """Training and validation dataset."""

    def __init__(self, data_list, classes, **kwargs):
        super().__init__(classes)

        class_counts = {class_name: 0 for class_name in classes}
        for _, class_name in data_list:
            class_counts[class_name] += 1

        sample_weights = [
            len(data_list) / class_counts[class_name] for _, class_name in data_list
        ]

        self.data = data_list
        self.sample_weights = sample_weights

    def __getitem__(self, idx):
        wav, label, audio_path = super().__getitem__(idx)
        '''
        # _silence_ audios are longer than 1 sec.
        if label == self.class2index["_silence_"]:
            random_offset = randint(0, len(wav) - 16000)
            wav = wav[random_offset : random_offset + 16000]
        '''
        return wav, label, audio_path


class SpeechCommandsTestingDataset(SpeechCommandsBaseDataset):
    """Testing dataset."""

    def __init__(self, **kwargs):
        super().__init__()

        self.data = [
            (class_dir.name, audio_path)
            for class_dir in Path(kwargs["speech_commands_test_root"]).iterdir()
            if class_dir.is_dir()
            for audio_path in class_dir.glob("*.wav")
        ]
