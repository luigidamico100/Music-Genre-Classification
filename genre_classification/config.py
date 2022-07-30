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

##########################################################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")
