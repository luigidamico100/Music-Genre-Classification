from src.paths import (
    path_raw_wav_original,
    path_raw_images_original,
    path_annotation_wav_original,
    path_annotation_images_original,
)
import os
import pandas as pd


def create_df_annotation(path_orig):
    list_dict = []
    sample_dict = {}
    for genre in os.listdir(path_orig):
        genre_fullpath = os.path.join(path_orig, genre)
        for filename in os.listdir(genre_fullpath):
            fullpath = os.path.join(genre_fullpath, filename)
            dict_sample = {'genre': genre, 'path': fullpath, 'filename': filename}
            list_dict.append(dict_sample)

    df_annotation = pd.DataFrame(list_dict)
    return df_annotation


def main():
    df_annotation_wav = create_df_annotation(path_raw_wav_original)
    df_annotation_images = create_df_annotation(path_raw_images_original)

    print(f'Writing annotation .wav files to: {path_annotation_wav_original}')
    df_annotation_wav.to_csv(path_annotation_wav_original)
    print(f'Writing annotation images files to: {path_annotation_images_original}')
    df_annotation_images.to_csv(path_annotation_images_original)


if __name__ == '__main__':
    main()
