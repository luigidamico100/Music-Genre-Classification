from genre_classification.config import (
    path_raw_wav_original,
    path_raw_images_original,
    path_annotation_original,
)
import os
import pandas as pd


def create_df_annotation(path_orig):
    list_dict = []
    for genre in os.listdir(path_orig):
        genre_fullpath = os.path.join(path_orig, genre)
        for filename in os.listdir(genre_fullpath):
            fullpath = os.path.join(genre_fullpath, filename)
            dict_sample = {'genre': genre, 'path': fullpath, 'filename': filename}
            list_dict.append(dict_sample)

    df_annotation = pd.DataFrame(list_dict)
    return df_annotation


def merge_annotations(df_annotation_wav, df_annotation_images):
    list_dict = []

    for idx, df_annotation_wav_row in df_annotation_wav.iterrows():
        wav_filename = df_annotation_wav_row['filename']
        dict_sample = {'genre': df_annotation_wav_row['genre'],
                       'wav_path': df_annotation_wav_row['path'],
                       'wav_filename': wav_filename}

        # Finding the associated image file
        wav_filename_split = wav_filename.split('.')
        image_filename = wav_filename_split[0] + wav_filename_split[1] + '.png'
        df_annotation_images_row = df_annotation_images[df_annotation_images['filename']==image_filename]
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


def main():
    df_annotation_wav = create_df_annotation(path_raw_wav_original)
    df_annotation_images = create_df_annotation(path_raw_images_original)

    df_annotation = merge_annotations(df_annotation_wav, df_annotation_images)

    print(f'Writing annotation files to: {path_annotation_original}')
    df_annotation.to_csv(path_annotation_original)

if __name__ == '__main__':
    main()
