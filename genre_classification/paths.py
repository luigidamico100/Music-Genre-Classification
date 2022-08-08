import os
from pathlib import Path
import torch

################################################### Paths #################################################
project_root_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)

# Raw data paths
path_raw_wav_original = os.path.join(project_root_path, 'data', 'raw', 'genres_original')
path_raw_images_original = os.path.join(project_root_path, 'data', 'raw', 'images_original')
path_raw_features_3_sec = os.path.join(project_root_path, 'data', 'raw', 'features_3_sec.csv')
path_raw_features_30_sec = os.path.join(project_root_path, 'data', 'raw', 'features_30_sec.csv')

# Annotation dataframe
path_annotation_original = os.path.join(project_root_path, 'data', 'interim', 'annotation.csv')

# Experiment
experiment_name = 'trial'
path_training_experiments = os.path.join(project_root_path, 'models', 'experiments')

def get_path_experiment(experiment_name, file_type=None):
    
    if not file_type:
        return os.path.join(path_training_experiments, experiment_name)
    
    assert file_type in ['df_training_data', 'training_plot', 'best_model']
    
    if file_type=='df_training_data':
        file_name = 'df_training_data.csv'
    elif file_type=='training_plot':
        file_name = 'training_plot.jpg'
    elif file_type=='best_model':
        file_name = 'best_model.pth'
    
    return os.path.join(path_training_experiments, experiment_name, file_name)
##########################################################################################################