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
path_class_to_genre_map = os.path.join(project_root_path, 'data', 'interim', 'class_to_genre_map.pkl')
path_genre_to_class_map = os.path.join(project_root_path, 'data', 'interim', 'genre_to_class_map.pkl')


# Experiment
path_training_experiments = os.path.join(project_root_path, 'models', 'experiments')

def get_path_experiment(experiment_name, file_type, overwrite_existing_experiment=True):
    
    def create_dir(path, overwrite_existing_experiment=False):
        try:
            os.mkdir(path)
        except FileExistsError:
            if overwrite_existing_experiment:
                pass
            else:
                raise FileExistsError()
    
    path_folder = os.path.join(path_training_experiments, experiment_name)
    path_folder_training = os.path.join(path_folder, 'training')
    path_folder_evaluation = os.path.join(path_folder, 'evaluation')
    path_folder_embeddings = os.path.join(path_folder, 'embeddings')
    
    create_dir(path_folder, overwrite_existing_experiment)
    create_dir(path_folder_training, overwrite_existing_experiment)
    create_dir(path_folder_evaluation, overwrite_existing_experiment)
    create_dir(path_folder_embeddings, overwrite_existing_experiment)
    
    admitted_file_type = ['df_training_history', 'training_plot', 'best_model', 'training_log', 'json', 
                          'df_conf_matrix_val', 'df_conf_matrix_norm_val', 'metrics_val',
                          'df_conf_matrix_test', 'df_conf_matrix_norm_test', 'metrics_test',
                          'df_conf_matrix_train', 'df_conf_matrix_norm_train', 'metrics_train',
                          'df_conf_matrix_all', 'df_conf_matrix_norm_all', 'metrics_all',
                          'df_embeddings_all', 'df_genres_all',
                          'df_embeddings_train', 'df_genres_train',
                          'df_embeddings_val', 'df_genres_val',
                          'df_embeddings_test', 'df_genres_test']
    assert file_type in admitted_file_type
    
    # ---------- training ---------- 
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
        
    # ---------- evaluation ---------- 
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
    elif file_type=='metrics_train':
        file_name = 'metrics_train.txt'
        path_file = os.path.join(path_folder_evaluation, file_name)
    elif file_type=='df_conf_matrix_train':
        file_name = 'df_confusion_matrix_train.csv'
        path_file = os.path.join(path_folder_evaluation, file_name)
    elif file_type=='df_conf_matrix_norm_train':
        file_name = 'df_confusion_matrix_norm_train.csv'
        path_file = os.path.join(path_folder_evaluation, file_name)
    elif file_type=='metrics_all':
        file_name = 'metrics_all.txt'
        path_file = os.path.join(path_folder_evaluation, file_name)
    elif file_type=='df_conf_matrix_all':
        file_name = 'df_confusion_matrix_all.csv'
        path_file = os.path.join(path_folder_evaluation, file_name)
    elif file_type=='df_conf_matrix_norm_all':
        file_name = 'df_confusion_matrix_norm_all.csv'
        path_file = os.path.join(path_folder_evaluation, file_name)
        
    # ---------- embeddings ---------- 
    elif file_type=='df_embeddings_val':
        file_name = 'df_embeddings_val.csv'
        path_file = os.path.join(path_folder_embeddings, file_name)
    elif file_type=='df_genres_val':
        file_name = 'df_genres_val.csv'
        path_file = os.path.join(path_folder_embeddings, file_name)
    elif file_type=='df_embeddings_test':
        file_name = 'df_embeddings_test.csv'
        path_file = os.path.join(path_folder_embeddings, file_name)
    elif file_type=='df_genres_test':
        file_name = 'df_genres_test.csv'
        path_file = os.path.join(path_folder_embeddings, file_name)
    elif file_type=='df_embeddings_train':
        file_name = 'df_embeddings_train.csv'
        path_file = os.path.join(path_folder_embeddings, file_name)
    elif file_type=='df_genres_train':
        file_name = 'df_genres_train.csv'
        path_file = os.path.join(path_folder_embeddings, file_name)
    elif file_type=='df_embeddings_all':
        file_name = 'df_embeddings_all.csv'
        path_file = os.path.join(path_folder_embeddings, file_name)
    elif file_type=='df_genres_all':
        file_name = 'df_genres_all.csv'
        path_file = os.path.join(path_folder_embeddings, file_name)
    
    return path_file
##########################################################################################################