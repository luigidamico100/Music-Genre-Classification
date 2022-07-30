from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch


class GTZANDataset(Dataset):

    def __init__(self,
                 annotations_file_path,
                 transformation=None,
                 target_sample_rate=None,
                 num_samples=None,
                 device=None):
        self.annotations = pd.read_csv(annotations_file_path, index_col=0)
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.transformation = transformation.to(device)
        self.genre_to_class = {genre: idx for idx, genre in enumerate(self.annotations['genre'].unique())}
        self.class_to_genre = {self.genre_to_class[genre]: genre for genre in self.genre_to_class}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample_path = self.annotations.iloc[idx]['wav_path']
        genre = self.annotations.iloc[idx]['genre']
        label = self.genre_to_class[genre]
        signal, sr = torchaudio.load(sample_path)
        signal = self.resample(signal, sr)
        signal = self.cut(signal)
        signal = self.right_pad(signal)
        signal = signal.to(self.device)
        signal = self.transformation(signal)
        return signal, label

    def resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def cut(self, signal):
        length_signal = signal.shape[1]
        if length_signal > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def right_pad(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
