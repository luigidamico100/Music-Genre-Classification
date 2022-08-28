Music-Genre-Classification
==============================

Music genre classification using [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), a dataset containing 1000 labeled audio files of 10 different genres. 

The model takes as input the Mel Spectrogram of the song and analyzes the image using a Convolutional Neural Network (CNN). The output is a vector containing the score associated to each genre.


# Dataset

The dataset contains 1000 labeled audio files (belonging to famous songs) all having a length of 30 seconds of 10 different genres. The genres are:

 - Blues
 - Classical
 - Country
 - Disco
 - HipHop
 - Jazz
 - Metal
 - Pop
 - Raggae
 - Rock



The dataset can be downloaded [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). Place the content in `data/raw/` folder. For this project only the folder `genres_original` will be used. It contains the audio files in .wav format.







# Installation

Place in the current directory. Clone the current conda environment or install package directly from requirements, then, install the project package. 

### Cloning the code environment

The conda environment has been exported using the command

    conda env export > environment.yml

Type the following instruction to clone the environment

    conda env create -f environment.yml


###  Install from requirements
The requirements has been generated using

    pip freeze > requirements.txt


Install package requirements using pip 

    pip install -r requirements.txt

or conda

    conda create --name <env_name> --file requirements.txt



### Install the project package
The project package can be install using

    pip install -e .


The project folder [genre_classification](genre_classification) is installed as a python package and it can be now referenced anywhere. 



# API service

Once an experiment is executed and a trained model is ready in the [experiments](models/experiments) folder (follow the guide in [genre_classification](genre_classification)), the API service can be used. The API service is used to make inference on a new and unlabeled wav file located in `/data/external` folder.

Start the server by running

    uvicorn app:app --reload

Run your browser to test the service API.

Go to the address 

    http://127.0.0.1:8000

or 

    http://127.0.0.1:8000/docs



Project Organization
------------



    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │   └── external       <- External data to use for inference. Wav files without label
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   ├── 1.0-exploration.ipynb
    │   ├── 2.0-google-colab.ipynb
    │   └── 3.0-visualize_embeddings.ipynb
    │
    ├── requirements.txt  
    │
    ├── setup.py           <- makes project pip installable so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py <- Generate the final dataset used for modelling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── config.py
    │   │   ├── cnn.py
    │   │   ├── dataset.py
    │   │   ├── train.py
    │   │   ├── evaluate.py
    │   │   ├── embeddings.py
    │   │   └── inference.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── app.py              <- FastAPI service



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
