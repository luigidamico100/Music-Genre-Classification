from torch.utils.data import Dataset, DataLoader
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

def create_data_loader(dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == "__main__":
    from genre_classification.config import path_annotation_original, device

    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050 * 30 # Check this!
    SAMPLE_LEN = 661794
    BATCH_SIZE = 32


    # ANNOTATIONS_FILE = "/home/valerio/datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
    # AUDIO_DIR = "/home/valerio/datasets/UrbanSound8K/audio"
    # SAMPLE_RATE = 22050
    # NUM_SAMPLES = 22050

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    dataset = GTZANDataset(path_annotation_original,
                           mel_spectrogram,
                           SAMPLE_RATE,
                           NUM_SAMPLES,
                           device)

    dataloader = create_data_loader(dataset, batch_size=BATCH_SIZE)
    dataloader_it = iter(dataloader)
    dataloader_out = next(dataloader_it)

    print(f"There are {len(dataset)} samples in the dataset.")
    signal, label = dataset[0]
    print(signal.shape)
    print(label)
    print(dataloader_out[0].shape)
    print(dataloader_out[1].shape)