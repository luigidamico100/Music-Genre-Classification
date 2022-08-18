from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import torch
import random
import pickle
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


def read_wav_and_preprocess(wav_file_path,
                            wav_to_spec_transformation,
                            target_sample_rate,
                            num_samples,
                            training=False,
                            verbose_sample_wasting=False,
                            device='cpu'):
    
    
    def resample(signal, sr, target_sample_rate):
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def mix_down(signal):
        signal = torch.mean(signal, dim=0, keepdim=False)
        return signal
    
    def adjust_audio_len(signal, num_samples):
        random_index = random.randint(0, len(signal) - num_samples - 1)
        signal = signal[random_index : random_index + num_samples]
        return signal
    
    def get_signal_chunks(signal, num_samples, target_sample_rate):
        num_chunks = len(signal) // num_samples
        len_pre = len(signal)
        signal = signal[:num_chunks*num_samples]
        len_post = len(signal)
        signal = torch.reshape(signal, (num_chunks, num_samples))
        if verbose_sample_wasting:
            print(f'Losed {len_pre - len_post} samples ({((len_pre-len_post)/target_sample_rate):.1f}) sec')
        return signal
    
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
        
    
    signal, sr = torchaudio.load(wav_file_path)
    signal = signal.to(device)
    signal = resample(signal, sr, target_sample_rate)
    signal = mix_down(signal)
    if training:
        # signal = signal.unsqueeze(0)
        # #signal = signal.to('cpu')
        # signal = self.augmentation(signal)
        # #signal = signal.to(self.device)
        # signal = signal.squeeze(0)
        
        signal = adjust_audio_len(signal, num_samples)
    else:
        signal = get_signal_chunks(signal, num_samples, target_sample_rate)
        
    signal = wav_to_spec_transformation(signal)
    
    return signal, sr




class GTZANDataset(Dataset):

    def __init__(self,
                 path_annotations_file,
                 path_class_to_genre_map,
                 path_genre_to_class_map,
                 n_examples='all',
                 target_sample_rate=None,
                 device=None,
                 folds=[0, 1, 2, 3, 4, 5],
                 training=True,
                 chunks_len_sec=7.,
                 verbose_sample_wasting=False,
                 return_wav_filename=False,
                 wav_to_spec_transformation=None,
                 ):

        self.annotations = pd.read_csv(path_annotations_file, index_col=0)
        with open(path_class_to_genre_map, 'rb') as f:
            self.class_to_genre_map = pickle.load(f)
        with open(path_genre_to_class_map, 'rb') as f:
            self.genre_to_class_map = pickle.load(f)

        self.genres = list(self.genre_to_class_map.keys())
        self.annotations = self.annotations[self.annotations['fold'].isin(folds)]
        if n_examples!='all':
            self.annotations = self.annotations.sample(n_examples, random_state=42)
        self.training = training
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.num_samples = int(self.target_sample_rate * chunks_len_sec)
        #self.transformation = transformation.to(device)
        self.verbose_sample_wasting = verbose_sample_wasting
        
        self.wav_to_spec_transformation = wav_to_spec_transformation
        
        self.return_wav_filename = return_wav_filename

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        '''
        self.training == True
            return torch.Tensor of shape (f, t).
                f = n_fft
                t = (self.num_samples) / hop_length
        self.training == False
            return torch.Tensor of shape (c, f, t)
        
        c = n chunks
        f = n frequency bins
        t = n time bins
        
        When used with dataloader, batch dimension b has to be added!
        '''
        sample_path = self.annotations.iloc[idx]['wav_path']
        genre = self.annotations.iloc[idx]['genre']
        label = self.genre_to_class_map[genre]
        
        signal, sr = read_wav_and_preprocess(wav_file_path=sample_path,
                                             wav_to_spec_transformation=self.wav_to_spec_transformation,
                                             target_sample_rate=self.target_sample_rate,
                                             num_samples=self.num_samples,
                                             training=self.training,
                                             verbose_sample_wasting=self.verbose_sample_wasting,
                                             device=self.device)
        
        if self.return_wav_filename:
            wav_filename = self.annotations.iloc[idx]['wav_filename']
            return signal, label, wav_filename
        return signal, label



def get_wav_to_spec_transformation(mel_spectrogram_params, target_sample_rate, device):
    
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            normalized=False,
            **mel_spectrogram_params,).to(device)
        amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)
        
        wav_to_spec_transformation = [mel_spectrogram, amplitude_to_db]
        return Compose(wav_to_spec_transformation)
    
    
def create_data_loader(set_='train', batch_size=128, mel_spectrogram_params=None, **dataset_kwargs):
    assert set_ in ['train', 'val', 'test', 'all']
    
    target_sample_rate = dataset_kwargs['target_sample_rate']
    chunks_len_sec = dataset_kwargs['chunks_len_sec']
    training = dataset_kwargs['training']
    
    if set_ == 'train':
        folds = list(range(0, 14))
    elif set_ == 'val':
        folds = [14, 15, 16]
    elif set_ == 'test':
        folds = [17, 18, 19]
    elif set_ == 'all':
        folds = list(range(0,20))
            
    if not training:
        avg_n_samples_signal = 661794       # Adjust this!!!
        num_chunks = (avg_n_samples_signal / target_sample_rate) / chunks_len_sec
        batch_size = math.floor(batch_size / num_chunks)
        
        
    wav_to_spec_transformation = get_wav_to_spec_transformation(mel_spectrogram_params, dataset_kwargs['target_sample_rate'], dataset_kwargs['device'])
    
    dataset = GTZANDataset(folds=folds, 
                           wav_to_spec_transformation=wav_to_spec_transformation,
                           **dataset_kwargs,
                           )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset


def create_inference_data_loader(wav_file_path, mel_spectrogram_params=None, **dataset_kwargs):
    target_sample_rate = dataset_kwargs['target_sample_rate']
    chunks_len_sec = dataset_kwargs['chunks_len_sec']
    training = dataset_kwargs['training']

    avg_n_samples_signal = 661794  # Adjust this!!!
    num_chunks = (avg_n_samples_signal / target_sample_rate) / chunks_len_sec

    dataset = GTZANDataset(mel_spectrogram_kwargs=mel_spectrogram_params,
                           **dataset_kwargs,
                           )
    dataloader = DataLoader(dataset, batch_size=1)
    return dataloader, dataset


def main():
    import numpy as np
    from genre_classification.paths import (
        path_annotations,
        path_class_to_genre_map,
        path_genre_to_class_map,
    )
    from genre_classification.models import config

    mel_spectrogram_params = {'n_fft': config.melspec_fft,
                              'hop_length': config.melspec_hop_length,
                              'n_mels': config.melspec_n_mels}
    dataloader, dataset = create_data_loader(set_='train',
                                             batch_size=config.batch_size,
                                             mel_spectrogram_params=mel_spectrogram_params,
                                             path_annotations_file=path_annotations,
                                             path_class_to_genre_map=path_class_to_genre_map,
                                             path_genre_to_class_map=path_genre_to_class_map,
                                             training=True,
                                             n_examples='all',
                                             target_sample_rate=config.sample_rate,
                                             chunks_len_sec=config.chunks_len_sec,
                                             device=config.device,)

    dataloader_it = iter(dataloader)
    dataloader_out = next(dataloader_it)

    print(f"There are {len(dataset)} samples in the dataset.")
    signal, label = dataset[0]
    print(signal.shape)
    print(label)
    print(dataloader_out[0].shape)
    print(dataloader_out[1].shape)

    sample = np.array(dataloader_out[0][0][0])
    
    
if __name__ == "__main__":
    main()
    
