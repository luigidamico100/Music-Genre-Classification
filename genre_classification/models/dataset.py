from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import torch
import random
import math
from torchaudio_augmentations import (
    RandomResizedCrop,
    RandomApply,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
    Compose,
)



class GTZANDataset(Dataset):

    def __init__(self,
                 annotations_file_path,
                 n_examples='all',
                 target_sample_rate=None,
                 device=None,
                 folds=[0, 1, 2, 3, 4, 5],
                 split='train',
                 chunks_len_sec=7.,
                 verbose_sample_wasting=False):

        self.annotations = pd.read_csv(annotations_file_path, index_col=0)
        self.annotations = self.annotations[self.annotations['fold'].isin(folds)]
        if n_examples!='all':
            self.annotations = self.annotations.sample(n_examples, random_state=42)
        assert split in ['train', 'val', 'test']
        self.split = split
        self.training = split=='train'
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.num_samples = int(self.target_sample_rate * chunks_len_sec)
        #self.transformation = transformation.to(device)
        self.genre_to_class = {genre: idx for idx, genre in enumerate(self.annotations['genre'].unique())}
        self.class_to_genre = {self.genre_to_class[genre]: genre for genre in self.genre_to_class}
        self.verbose_sample_wasting = verbose_sample_wasting
        if self.training:
            self._get_augmentations()
        
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64,
            normalized=False
        ).to(device)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        '''
        self.split == 'train'
            return torch.Tensor of shape (f, t).
                f = n_fft
                t = (self.num_samples) / hop_length
        self.split == 'val' or 'test'
            return torch.Tensor of shape (c, f, t)
        
        c = n chunks
        f = n frequency bins
        t = n time bins
        
        When used with dataloader, batch dimension b has to be added!
        '''
        sample_path = self.annotations.iloc[idx]['wav_path']
        genre = self.annotations.iloc[idx]['genre']
        label = self.genre_to_class[genre]
        signal, sr = torchaudio.load(sample_path)
        signal = signal.to(self.device)
        signal = self.resample(signal, sr)
        signal = self.mix_down(signal)
        #signal = self.adjust_audio_len(signal)
        # signal = self.cut(signal)
        # signal = self.right_pad(signal)
        if self.training:
            signal = signal.unsqueeze(0)
            #signal = signal.to('cpu')
            signal = self.augmentation(signal)
            #signal = signal.to(self.device)
            signal = signal.squeeze(0)
        else:
            signal = self.get_signal_chunks(signal)
        signal = self.mel_spectrogram(signal)
        signal = self.amplitude_to_db(signal)
        
        return signal, label
    
    def _get_augmentations_pytorch(self):
        from torch import nn
        transforms = nn.Sequential(
            torchaudio.transforms.Vol()
            )
        
        transforms = [
            RandomResizedCrop(n_samples=self.num_samples).to(self.device),
            RandomApply([PolarityInversion()], p=0.8).to(self.device),
            #RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3).to(self.device),
            RandomApply([Gain()], p=0.2).to(self.device),
            RandomApply([HighLowPass(sample_rate=22050)], p=0.8).to(self.device),
            RandomApply([Delay(sample_rate=22050)], p=0.5).to(self.device),
            #RandomApply([PitchShift(n_samples=self.num_samples, sample_rate=22050)], p=0.4).to(self.device),
            #RandomApply([Reverb(sample_rate=22050)], p=0.3).to(self.device),
        ]
        self.augmentation = Compose(transforms=transforms)

    def _get_augmentations(self):
        transforms = [
            RandomResizedCrop(n_samples=self.num_samples).to(self.device),
            RandomApply([PolarityInversion()], p=0.8).to(self.device),
            #RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3).to(self.device),
            RandomApply([Gain()], p=0.2).to(self.device),
            RandomApply([HighLowPass(sample_rate=22050)], p=0.8).to(self.device),
            RandomApply([Delay(sample_rate=22050)], p=0.5).to(self.device),
            #RandomApply([PitchShift(n_samples=self.num_samples, sample_rate=22050)], p=0.4).to(self.device),
            #RandomApply([Reverb(sample_rate=22050)], p=0.3).to(self.device),
        ]
        self.augmentation = Compose(transforms=transforms)


    def get_signal_chunks(self, signal):
        num_chunks = len(signal) // self.num_samples
        len_pre = len(signal)
        signal = signal[:num_chunks*self.num_samples]
        len_post = len(signal)
        signal = torch.reshape(signal, (num_chunks, self.num_samples))
        if self.verbose_sample_wasting:
            print(f'Losed {len_pre - len_post} samples ({((len_pre-len_post)/self.target_sample_rate):.1f}) sec')
        return signal
    
    def adjust_audio_len(self, signal):
        if self.split == 'train':
            random_index = random.randint(0, len(signal) - self.num_samples - 1)
            signal = signal[random_index : random_index + self.num_samples]
        else:
            num_chunks = len(signal) // self.num_samples
            len_pre = len(signal)
            signal = signal[:num_chunks*self.num_samples]
            len_post = len(signal)
            signal = torch.reshape(signal, (num_chunks, self.num_samples))
            if self.verbose_sample_wasting:
                print(f'Losed {len_pre - len_post} samples ({((len_pre-len_post)/self.target_sample_rate):.1f}) sec')
        return signal
        
    
    def resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def mix_down(self, signal):
        signal = torch.mean(signal, dim=0, keepdim=False)
        return signal

    # def cut_old(self, signal):
    #     length_signal = signal.shape[1]
    #     if length_signal > self.num_samples:
    #         signal = signal[:self.num_samples]
    #     return signal

    # def cut(self, signal):
    #     length_signal = len(signal)
    #     if length_signal > self.num_samples:
    #         random_index = random.randint(0, length_signal - self.num_samples - 1)
    #         signal = signal[random_index:random_index + self.num_samples]
    #     return signal

    # def right_pad(self, signal):
    #     length_signal = len(signal)
    #     if length_signal < self.num_samples:
    #         num_missing_samples = self.num_samples - length_signal
    #         last_dim_padding = (0, num_missing_samples)
    #         signal = torch.nn.functional.pad(signal, last_dim_padding)
    #     return signal


def create_data_loader(path_annotation_original,
                       n_examples=None,
                       target_sample_rate=None,
                       chunks_len_sec=7.,
                       device='cpu',
                       batch_size=64,
                       split='train',
                       verbose_sample_wasting=False):
    
    if split == 'train':
        folds = list(range(0, 14))
    else:
        avg_n_samples_signal = 661794       # Adjust this!!!
        num_chunks = (avg_n_samples_signal / target_sample_rate) / chunks_len_sec
        batch_size = math.floor(batch_size / num_chunks)
        if split == 'val':
            folds = [14, 15, 16]
        elif split == 'test':
            folds = [17, 18, 19]
        else:
            raise ValueError


    dataset = GTZANDataset(path_annotation_original,
                           n_examples=n_examples,
                           target_sample_rate=target_sample_rate,
                           chunks_len_sec=chunks_len_sec,
                           device=device,
                           folds=folds, 
                           split=split,
                           verbose_sample_wasting=verbose_sample_wasting)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset


            

#%%

if __name__ == "__main__":
    import numpy as np
    from genre_classification.paths import path_annotation_original
    from genre_classification.models.config import (
        device,
        batch_size,
        sample_rate,
        chunks_len_sec,
    )


    dataloader, dataset = create_data_loader(path_annotation_original,
                                             n_examples='all',
                                             target_sample_rate=sample_rate,
                                             chunks_len_sec=chunks_len_sec,
                                             device=device,
                                             batch_size=batch_size,
                                             split='test',)

    dataloader_it = iter(dataloader)
    dataloader_out = next(dataloader_it)

    print(f"There are {len(dataset)} samples in the dataset.")
    signal, label = dataset[0]
    print(signal.shape)
    print(label)
    print(dataloader_out[0].shape)
    print(dataloader_out[1].shape)

    sample = np.array(dataloader_out[0][0][0])
