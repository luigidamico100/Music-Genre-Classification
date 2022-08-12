#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 00:25:18 2022

@author: luigi
"""

import torch
import numpy as np
import pandas as pd
from genre_classification.models.dataset import create_data_loader
from genre_classification.paths import (
    path_annotations, 
    path_class_to_genre_map,
    path_genre_to_class_map,
    get_path_experiment
    )
from genre_classification.models import config
from genre_classification.models.evaluate import load_experiment


def get_embeddings(model, dataloader, device, class_to_genre_map):
    model.eval()
    embeddings_overall = torch.empty((0,)).to(device)
    target_overall = torch.empty((0,)).to(device)
    wav_filename_overall = np.empty(0,)
    with torch.no_grad():
        for input, target, wav_filename in dataloader:
            input, target = input.to(device), target.to(device)
            b, c, f, t = input.shape  # batchs, chunks, freq., time
            input = input.view(-1, f, t)    # (b, c, f, t) -> (b*c, f, t)
            embeddings = model(input)
            embeddings = embeddings.view(b, c, -1).mean(dim=1) # Check that

            embeddings_overall = torch.cat((embeddings_overall, embeddings))
            target_overall = torch.cat((target_overall, target))
            wav_filename_overall = np.concatenate((wav_filename_overall, wav_filename))

    embeddings_overall = np.array(embeddings_overall.cpu(), dtype=int)
    target_overall = np.array(target_overall.cpu(), dtype=int)
    
    df_embeddings = pd.DataFrame(embeddings_overall)
    df_genres = pd.DataFrame()
    df_genres['wav_filename'] = wav_filename_overall
    df_genres['genres'] = target_overall
    df_genres['genres'] = df_genres['genres'].map(class_to_genre_map)
    
    return df_embeddings, df_genres


def save_embeddings_data(dfs, experiment_name, set_='val'):
    
    df_embeddings, df_genres = dfs
    
    path_df_embeddings = get_path_experiment(experiment_name, file_type=f'df_embeddings_{set_}')
    path_df_genres = get_path_experiment(experiment_name, file_type=f'df_genres_{set_}')
    
    print(f'--------- {set_} ---------')
    print(f'Saving df_embeddings_{set_} to {path_df_embeddings}')
    df_embeddings.to_csv(path_df_embeddings, index=False, header=False, sep='\t')
    print(f'Saving df_genres_{set_} to {path_df_genres}')
    df_genres.to_csv(path_df_genres, index=False, sep='\t')
    print()
    
    
    
#%%

def main(config):
    parsed_params = config.parse_params(config, reason='evaluate')
    print(parsed_params)
    experiment_name = parsed_params['experiment_name']
    set_ = parsed_params['set']
    cnn, params = load_experiment(experiment_name, return_embeddings=True, device=config.device)
    
    n_examples = params['n_examples']
    chunks_len_sec = params['chunks_len_sec']
    
    
    mel_spectrogram_params = {'n_fft': config.melspec_fft,
                              'hop_length': config.melspec_hop_length,
                              'n_mels': config.melspec_n_mels}
    
    dataloader, dataset = create_data_loader(set_=set_,
                                             batch_size=config.batch_size,
                                             mel_spectrogram_params=mel_spectrogram_params,
                                             path_annotations_file=path_annotations,
                                             path_class_to_genre_map=path_class_to_genre_map,
                                             path_genre_to_class_map=path_genre_to_class_map,
                                             training=False,
                                             n_examples=n_examples,
                                             target_sample_rate=config.sample_rate,
                                             chunks_len_sec=chunks_len_sec,
                                             device=config.device,
                                             return_wav_filename=True)
    
    dfs = get_embeddings(cnn, dataloader, config.device, dataset.class_to_genre_map)
    
    
    save_embeddings_data(dfs, experiment_name, set_=set_)
    
    
if __name__=='__main__':
    main(config)


