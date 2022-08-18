#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 17:25:22 2022

@author: luigi
"""

import torch
import numpy as np
import pandas as pd
from genre_classification.paths import path_class_to_genre_map
from genre_classification.models.dataset import get_preprocessed_wav
from genre_classification.models import config
from genre_classification.models.evaluate import load_experiment


def predict(model, signal_chunks, class_to_genre_map):
    model.eval()
    with torch.no_grad():
        prediction = model(signal_chunks)
        # prediction = prediction.view(b, c, -1).mean(dim=1) # Check that
        prediction = prediction.mean(dim=0)
        predicted = torch.argmax(prediction)
        
    predicted_genre = class_to_genre_map[predicted.item()]
    genres = list(class_to_genre_map.values())
    df_prediction_proba = pd.DataFrame(index=genres, data=np.array(prediction), columns=['class_proba'])
    df_prediction_proba = df_prediction_proba.sort_values(by='class_proba', ascending=False)
            
    return predicted_genre, df_prediction_proba
            


parsed_params = config.parse_params(config, reason='inference')
print('----- Parsed params -----')
print(parsed_params)
print()

cnn, params = load_experiment(parsed_params, device=config.device)
print('----- Params -----')
print(params)
print()

wav_path = parsed_params['wav_path']
experiment_name = parsed_params['experiment_name']


mel_spectrogram_params = {'n_fft': params['melspec_fft'],
                          'hop_length': params['melspec_hop_length'],
                          'n_mels': params['melspec_n_mels']}

signal_chunks, class_to_genre_map = get_preprocessed_wav(wav_file_path=wav_path,
                                     path_class_to_genre_map=path_class_to_genre_map,
                                      mel_spectrogram_params=mel_spectrogram_params,
                                      target_sample_rate=config.sample_rate,
                                      chunks_len_sec=params['chunks_len_sec'],
                                      device=config.device)



cnn, params = load_experiment(parsed_params, device=config.device)

chunks_len_sec = params['chunks_len_sec']


predicted_genre, df_prediction_proba = predict(cnn, signal_chunks, class_to_genre_map)

print(predicted_genre)
print()
print(df_prediction_proba)
