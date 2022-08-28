#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 13:17:00 2022

@author: luigi
"""

# pip install fastapi uvicorn
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from genre_classification.models.inference import inference


class MakeAnInference(BaseModel):
    experiment_name: str
    wav_filename: str


app = FastAPI()


# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Music genre classification project!'}


# Route with a single parameter, returns the parameter within a message
# Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcom to Luigi project': f'{name}'}


# Expose the prediction functionality, make a prediction from the passed
# JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_banknote(data: MakeAnInference):
    data = data.dict()
    experiment_name = data['experiment_name']
    wav_filename = data['wav_filename']
    predicted_genre, df_prediction_proba = inference(experiment_name, wav_filename)

    return {'Predicted genre': f'{predicted_genre}'}


# Run the API with uvicorn
# Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# First param: filename
# Second param: object name
#uvicorn app:app --reload