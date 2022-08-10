#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 00:25:18 2022

@author: luigi
"""

import torch
import numpy as np
import argparse
import pandas as pd
from genre_classification.models.cnn import CNNNetwork
from genre_classification.models.dataset import create_data_loader
from genre_classification.paths import path_annotation_original, experiment_name, get_path_experiment
from genre_classification.models.config import (
    device,
    batch_size,
    sample_rate,
)
import json
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from genre_classification.models.evaluate import load_experiment, get_experiment_name


def get_embeddings(model, dataloader, device, class_to_genre_dict):
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
    df_genres['genres'] = df_genres['genres'].map(class_to_genre_dict)
    
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

def main(experiment_name):

    experiment_name = get_experiment_name(experiment_name)
    cnn, params = load_experiment(experiment_name, return_embeddings=True)
    
    n_examples = params['n_examples']
    chunks_len_sec = params['chunks_len_sec']
    
    
    val_dataloader, val_dataset = create_data_loader(path_annotation_original,
                                                     n_examples='all',
                                                     target_sample_rate=sample_rate,
                                                     chunks_len_sec=chunks_len_sec,
                                                     device=device,
                                                     batch_size=batch_size,
                                                     split='val',
                                                     return_wav_filename=True)
    
    test_dataloader, test_dataset = create_data_loader(path_annotation_original,
                                                     n_examples='all',
                                                     target_sample_rate=sample_rate,
                                                     chunks_len_sec=chunks_len_sec,
                                                     device=device,
                                                     batch_size=batch_size,
                                                     split='test',
                                                     return_wav_filename=True)    
    
    dfs_val = get_embeddings(cnn, val_dataloader, device, val_dataset.class_to_genre)
    dfs_test = get_embeddings(cnn, test_dataloader, device, test_dataset.class_to_genre)
    
    
    save_embeddings_data(dfs_val, experiment_name, set_='val')
    save_embeddings_data(dfs_test, experiment_name, set_='test')
    
    
if __name__=='__main__':
    main(experiment_name)


