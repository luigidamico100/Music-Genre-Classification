from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import torch
import random


class GTZANDataset(Dataset):

    def __init__(self,
                 annotations_file_path,
                 n_samples=None,
                 transformation=None,
                 target_sample_rate=None,
                 num_samples=None,
                 device=None,
                 folds=[0, 1, 2, 3, 4, 5]):

        self.annotations = pd.read_csv(annotations_file_path, index_col=0)
        self.annotations = self.annotations[self.annotations['fold'].isin(folds)]
        if n_samples:
            self.annotations = self.annotations.sample(n_samples, random_state=42)
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
        signal = signal.to(self.device)
        signal = self.resample(signal, sr)
        signal = self.mix_down(signal)
        signal = self.cut(signal)
        signal = self.right_pad(signal)
        signal = self.transformation(signal)
        return signal, label

    def resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def mix_down(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def cut_old(self, signal):
        length_signal = signal.shape[1]
        if length_signal > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def cut(self, signal):
        length_signal = signal.shape[1]
        if length_signal > self.num_samples:
            random_index = random.randint(0, length_signal - self.num_samples - 1)
            signal = signal[:, random_index:random_index + self.num_samples]
        return signal

    def right_pad(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal


def create_data_loader(path_annotation_original,
                       n_samples=None,
                       transformation=None,
                       target_sample_rate=None,
                       num_samples=None,
                       device='cpu',
                       batch_size=64,
                       usage='train'):
    if usage == 'train':
        folds = list(range(0, 14))
    elif usage == 'val':
        folds = [14, 15, 16]
    elif usage == 'test':
        folds = [17, 18, 19]
    else:
        raise ValueError

    dataset = GTZANDataset(path_annotation_original,
                           n_samples=n_samples,
                           transformation=transformation,
                           target_sample_rate=target_sample_rate,
                           num_samples=num_samples,
                           device=device,
                           folds=folds)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset


if __name__ == "__main__":
    import numpy as np
    from genre_classification.paths import path_annotation_original
    from genre_classification.models.config import (
        device,
        batch_size,
        sample_rate,
        num_samples,
    )

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
        normalized=False
    )

    dataloader, dataset = create_data_loader(path_annotation_original,
                                             n_samples=None,
                                             transformation=mel_spectrogram,
                                             target_sample_rate=sample_rate,
                                             num_samples=num_samples,
                                             device=device,
                                             batch_size=batch_size,
                                             usage='val')

    dataloader_it = iter(dataloader)
    dataloader_out = next(dataloader_it)

    print(f"There are {len(dataset)} samples in the dataset.")
    signal, label = dataset[0]
    print(signal.shape)
    print(label)
    print(dataloader_out[0].shape)
    print(dataloader_out[1].shape)

    sample = np.array(dataloader_out[0][0][0])
