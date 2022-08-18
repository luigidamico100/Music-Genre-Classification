from genre_classification.paths import (
    path_raw_wav_original,
    path_raw_images_original,
    path_annotations,
    path_genre_to_class_map,
    path_class_to_genre_map,
)
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pickle


def create_df_annotation(path_orig):
    """
    The create_df_annotation function creates a dataframe of the annotations for each song.
    The function takes in one argument, which is the path to the original dataset.
    It returns two objects: df_annotation and genres.
    df_annotation is a pandas dataframe that contains three columns: genre, filename, and path.
    genres is a list of all genres in our dataset.

    :param path_orig: Specify the path to the folder containing all of the genre folders
    :return: A dataframe with the following columns: genre, filename, and path
    :doc-author: Trelent
    """
    """
    The create_df_annotation function creates a dataframe of the annotations for each song.
    The function takes in one argument, which is the path to the original dataset. 
    It returns two objects: df_annotation and genres. 
    df_annotation is a pandas dataframe that contains three columns: genre, filename, and path. 
    genres is a list of all genres in our dataset.
    
    :param path_orig: Specify the path to the folder containing all of the genre folders
    :return: A dataframe with the following columns:
    :doc-author: Trelent
    """
    list_dict = []
    genres = []
    for genre in os.listdir(path_orig):
        if genre == '.DS_Store':
            continue
        genres.append(genre)
        genre_fullpath = os.path.join(path_orig, genre)
        for filename in os.listdir(genre_fullpath):
            if filename == '.DS_Store':
                continue
            fullpath = os.path.join(genre_fullpath, filename)
            dict_sample = {'genre': genre, 'path': fullpath, 'filename': filename}
            list_dict.append(dict_sample)

    df_annotation = pd.DataFrame(list_dict)
    return df_annotation, genres


def create_class_genre_maps(genres):
    genres.sort()
    genre_to_class_map = {genre: idx for idx, genre in enumerate(genres)}
    class_to_genre_map = {genre_to_class_map[genre]: genre for genre in genre_to_class_map}
    return class_to_genre_map, genre_to_class_map


def merge_annotations(df_annotation_wav, df_annotation_images):
    """
    The merge_annotations function takes in two dataframes: df_annotation_wav and df_annotation_images.
    The first dataframe contains the annotations for the wav files, while the second contains annotations for
    the images. The function then merges these two dataframes together based on their filenames, which are
    extracted from each row of each respective dataframe. This is done to ensure that there is a one-to-one
    correspondence between wav files and their associated images.

    :param df_annotation_wav: Specify the dataframe containing the wav filenames and their paths
    :param df_annotation_images: Find the image file associated with a wav file
    :return: A dataframe with the following columns:
    :doc-author: Trelent
    """
    list_dict = []

    for idx, df_annotation_wav_row in df_annotation_wav.iterrows():
        wav_filename = df_annotation_wav_row['filename']
        dict_sample = {'genre': df_annotation_wav_row['genre'],
                       'wav_path': df_annotation_wav_row['path'],
                       'wav_filename': wav_filename}

        # Finding the associated image file
        wav_filename_split = wav_filename.split('.')
        image_filename = wav_filename_split[0] + wav_filename_split[1] + '.png'
        df_annotation_images_row = df_annotation_images[df_annotation_images['filename'] == image_filename]
        if len(df_annotation_images_row) == 1:
            dict_sample['image_path'] = df_annotation_images_row['path'].item()
            dict_sample['image_filename'] = df_annotation_images_row['filename'].item()
        elif len(df_annotation_images_row) == 0:
            dict_sample['image_path'] = None
            dict_sample['image_filename'] = None
            print(f'There is no image for wav filename: {wav_filename}')
        elif len(df_annotation_images_row) > 1:
            dict_sample['image_path'] = None
            dict_sample['image_filename'] = None
            print(f'There are more than one image for wav filename: {wav_filename}')

        list_dict.append(dict_sample)

    df_annotation = pd.DataFrame(list_dict)
    df_annotation.index.name = 'index'
    return df_annotation


def add_fold_column(df_annotation, val_size=.2, test_size=.2):
    X = df_annotation.drop('genre', axis=1)
    y = df_annotation['genre']

    skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
    df_annotation['fold'] = -1

    for fold, (train_idxs, test_idxs) in enumerate(skf.split(X, y)):
        df_annotation.loc[df_annotation.index[test_idxs], 'fold'] = fold

    return df_annotation


def main():
    df_annotation_wav, genres = create_df_annotation(path_raw_wav_original)
    df_annotation_images, _ = create_df_annotation(path_raw_images_original)
    class_to_genre_map, genre_to_class_map = create_class_genre_maps(genres)
    df_annotation = merge_annotations(df_annotation_wav, df_annotation_images)
    df_annotation = add_fold_column(df_annotation)

    # jazz.00054.wav file cannot be opened
    df_annotation = df_annotation[df_annotation['wav_filename'] != 'jazz.00054.wav']

    print(f'Writing df_annotation to: {path_annotations}')
    df_annotation.to_csv(path_annotations)
    
    print(f'Writing genre_to_class_map to {path_genre_to_class_map}')
    with open(path_genre_to_class_map, 'wb') as outfile:
        pickle.dump(genre_to_class_map, outfile)
        
    print(f'Writing class_to_genre_map to {path_class_to_genre_map}')
    with open(path_class_to_genre_map, 'wb') as outfile:
        pickle.dump(class_to_genre_map, outfile)


if __name__ == '__main__':
    main()
