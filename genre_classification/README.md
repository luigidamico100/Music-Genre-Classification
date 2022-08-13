# Paths

The scripts interact between them using files which can used as input or as output for the scripts. 
The file paths are stored in `genre_classification/paths.py` file. In the current configuration the variable are :

 

    - path_raw_wav_original = genre_classification/data/raw/genres_original
    - path_raw_images_original = genre_classificaiton/data/raw/images_original
    - path_raw_features_3_sec = genre_classification/data/raw/features_3_sec.csv
    - path_raw_features_30_sec = genre_classification/data/raw/features_30_sec.csv
    
    - path_annotations = genre_classification/data/processed/annotations.csv
    - path_class_to_genre_map = genre_classification/data/processed/class_to_genre_map.pkl
    - path_genre_to_class_map = genre_classification/data/processed/genre_to_class_map.pkl
    
    - path_training_experiments = genre_classification/models/experiments
    
The paths above contain the following things:

 - `path_raw_wav_original`: ....
 - `path_raw_images_original`
 - `path_raw_features_3_sec`
 - `path_raw_features_30_sec`

 - `path_annotations`
 - `path_class_to_genre_map`
 - `path_genre_to_class_map`
 
 - `path_training_experiments`



# Pipeline

## Generate the dataset

To generate the the annotation dataset used for the modelling, run

    python genre_classification/data/make_dataset.py
 
It takes as input the wav and images in `path_raw_wav_original` and `path_raw_images_original` a output the annotations file in `path_annotations`. This scripts divide the examples in different folds, with a total of `20` folds (`0..19`) in a stratified fashion i.e. in each folder, there is the same number of examples of different class (genre)

## Traing and evaluation


### Config file

The file in `model/config.py` containts parameters for both training and evaluation of the model. Some of these parameters can be passed as argument when a script is called. Passing parameters in this way have priority over the ones listed in the `config.py` file.

### Dataset
The script provides utility functions to the dataset and dataloader creation. Three set can be identified

 - `Training set`: identified by the first `14` folds. It constitutes the `70%` of the entire dataset.
 - `Validation set`: identified by the folds `14, 15, 16`. It constitutes the `15%` of the entire dataset.
 - `Test set`: identified by the folds `17, 18, 19`. It constitutes the `15%` of the entire dataset.

### Train the model
To train the model run

    python genre_classification/model/train.py --experiment_name {my_experiment}

You can see all the parameters that can be passed to a script by typing

    python genre_classification/model/train.py --help

This script take as input the annotation file in `path_annotations` and execute the training using `Training set` and evaluate the performance on the `Validation set`.


### Evaluate the model


