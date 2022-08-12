#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 23:43:09 2022

@author: luigi
"""

import torch
import numpy as np
import pandas as pd
from genre_classification.models.cnn import CNNNetwork
from genre_classification.models.dataset import create_data_loader
from genre_classification.paths import (
    path_annotations, 
    path_class_to_genre_map,
    path_genre_to_class_map,
    get_path_experiment
    )
from genre_classification.models import config
import json
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score



def evaluate(model, dataloader, device):
    model.eval()
    prediction_overall = torch.empty((0,)).to(device)
    target_overall = torch.empty((0,)).to(device)
    with torch.no_grad():
        for input, target in dataloader:
            input, target = input.to(device), target.to(device)
            b, c, f, t = input.shape  # batchs, chunks, freq., time
            input = input.view(-1, f, t)    # (b, c, f, t) -> (b*c, f, t)
            prediction = model(input)
            prediction = prediction.view(b, c, -1).mean(dim=1) # Check that

            _, predicted = torch.max(prediction.data, dim=1)

            prediction_overall = torch.cat((prediction_overall, predicted))
            target_overall = torch.cat((target_overall, target))

    prediction_overall = np.array(prediction_overall.cpu(), dtype=int)
    target_overall = np.array(target_overall.cpu(), dtype=int)
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(target_overall, prediction_overall)
    metrics['confusion matrix'] = confusion_matrix(target_overall, prediction_overall)
    metrics['confusion matrix norm'] = confusion_matrix(target_overall, prediction_overall, normalize='true')
    metrics['f1 score'] = f1_score(target_overall, prediction_overall, average='weighted')

    return metrics


def save_evaluation_data(metrics, genres, experiment_name, set_='val'):
    
    path_df_conf_matrix = get_path_experiment(experiment_name, file_type=f'df_conf_matrix_{set_}')
    path_df_conf_matrix_norm = get_path_experiment(experiment_name, file_type=f'df_conf_matrix_norm_{set_}')
    path_metrics = get_path_experiment(experiment_name, file_type=f'metrics_{set_}')
    
    print(f'--------- {set_} ---------')
    print(f"Accuracy = {metrics['accuracy']}")
    print(f"f1 score = {metrics['f1 score']}")
    print(f"Confusion matrix = \n {metrics['confusion matrix']}")
    print()
    
    print(f'Saving metrics_{set_} to {path_metrics}')
    metrics_text = f"Accuracy = {metrics['accuracy']:.3f}\nf1 score = {metrics['f1 score']:.3f}"
    with open(path_metrics, 'w') as f:
        f.write(metrics_text)
        
    print(f'Saving df_conf_matrix_{set_} to {path_df_conf_matrix}')
    df_conf_matrix = pd.DataFrame(data=metrics['confusion matrix'], columns=genres, index=genres)
    df_conf_matrix.to_csv(path_df_conf_matrix)
    
    print(f'Saving df_conf_matrix_norm_{set_} to {path_df_conf_matrix_norm}')
    df_conf_matrix = pd.DataFrame(data=metrics['confusion matrix norm'], columns=genres, index=genres)
    df_conf_matrix.to_csv(path_df_conf_matrix_norm)
    
    print()
    

def load_experiment(experiment_name, return_embeddings=False, device='cpu'):
    path_best_model = get_path_experiment(experiment_name, file_type='best_model')
    path_params = get_path_experiment(experiment_name, file_type='json')
    
    state_dict = torch.load(path_best_model)
    model = CNNNetwork(return_embeddings=return_embeddings)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    with open(path_params) as json_file:
        params = json.load(json_file)
        
    return model, params
    
    
#%%

def main(config):

    parsed_params = config.parse_params(config, reason='evaluate')
    print(parsed_params)
    experiment_name = parsed_params['experiment_name']
    set_ = parsed_params['set']
    cnn, params = load_experiment(experiment_name, device=config.device)
    
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
                                             device=config.device,)
    

    metrics = evaluate(cnn, dataloader, config.device)
    
    save_evaluation_data(metrics, dataset.genres, experiment_name, set_=set_)
    
    
if __name__=='__main__':
    main(config)


