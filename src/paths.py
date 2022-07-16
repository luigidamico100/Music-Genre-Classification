import os
from pathlib import Path

project_root_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)

# Raw data paths
path_raw_wav_original = os.path.join(project_root_path, 'data', 'raw', 'genres_original')
path_raw_images_original = os.path.join(project_root_path, 'data', 'raw', 'images_original')
path_raw_features_3_sec = os.path.join(project_root_path, 'data', 'raw', 'features_3_sec.csv')
path_raw_features_30_sec = os.path.join(project_root_path, 'data', 'raw', 'features_30_sec.csv')

# Annotation dataframes
path_annotation_wav_original = os.path.join(project_root_path, 'data', 'interim', 'annotation_wav.csv')
path_annotation_images_original = os.path.join(project_root_path, 'data', 'interim', 'annotation_images.csv')

