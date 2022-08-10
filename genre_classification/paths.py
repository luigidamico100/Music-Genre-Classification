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

def get_path_experiment(experiment_name, file_type, overwrite_existing_experiment=True):
    
    path_folder = os.path.join(path_training_experiments, experiment_name)
    path_folder_training = os.path.join(path_folder, 'training')
    path_folder_evaluation = os.path.join(path_folder, 'evaluation')
    
    try:
        os.mkdir(path_folder)
        os.mkdir(path_folder_training)
        os.mkdir(path_folder_evaluation)
    except FileExistsError:
        if overwrite_existing_experiment:
            pass
        else:
            raise FileExistsError
    
    admitted_file_type = ['df_training_history', 'training_plot', 'best_model', 'training_log', 'json', 
                          'df_conf_matrix_val', 'df_conf_matrix_norm_val', 'metrics_val',
                          'df_conf_matrix_test', 'df_conf_matrix_norm_test', 'metrics_test',]
    assert file_type in admitted_file_type
    
    if file_type=='df_training_history':
        file_name = 'df_training_history.csv'
        path_file = os.path.join(path_folder_training, file_name)
    elif file_type=='training_plot':
        file_name = 'training_plot.jpg'
        path_file = os.path.join(path_folder_training, file_name)
    elif file_type=='best_model':
        file_name = 'best_model.pth'
        path_file = os.path.join(path_folder, file_name)
    elif file_type=='training_log':
        file_name = 'training_log.log'
        path_file = os.path.join(path_folder_training, file_name)
    elif file_type=='json':
        file_name = 'params.json'
        path_file = os.path.join(path_folder, file_name)
    elif file_type=='metrics_val':
        file_name = 'metrics_val.txt'
        path_file = os.path.join(path_folder_evaluation, file_name)
    elif file_type=='df_conf_matrix_val':
        file_name = 'df_confusion_matrix_val.csv'
        path_file = os.path.join(path_folder_evaluation, file_name)
    elif file_type=='df_conf_matrix_norm_val':
        file_name = 'df_confusion_matrix_norm_val.csv'
        path_file = os.path.join(path_folder_evaluation, file_name)
    elif file_type=='metrics_test':
        file_name = 'metrics_test.txt'
        path_file = os.path.join(path_folder_evaluation, file_name)
    elif file_type=='df_conf_matrix_test':
        file_name = 'df_confusion_matrix_test.csv'
        path_file = os.path.join(path_folder_evaluation, file_name)
    elif file_type=='df_conf_matrix_norm_test':
        file_name = 'df_confusion_matrix_norm_test.csv'
        path_file = os.path.join(path_folder_evaluation, file_name)
    
    return path_file
##########################################################################################################