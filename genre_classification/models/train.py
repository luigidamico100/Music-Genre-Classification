import torch
import torchaudio
from torch import nn
from torchinfo import summary
from torchmetrics import Accuracy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
from genre_classification.models.cnn import CNNNetwork
from genre_classification.models.dataset import create_data_loader, GTZANDataset
from genre_classification.paths import path_annotation_original, path_experiment
from genre_classification.models.config import (
    device,
    batch_size,
    epochs,
    learning_rate,
    sample_rate,
    num_samples,
    debug
)


def train_single_epoch(model, dataloader, loss_fn, optimiser, device):
    model.train()
    accuracy = Accuracy().to(device)
    losses = []
    prediction_overall = torch.empty((0,)).to(device)
    target_overall = torch.empty((0,)).to(device)

    for input, target in dataloader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        _, predicted = torch.max(prediction.data, dim=1)
        prediction_overall = torch.cat((prediction_overall, predicted))
        target_overall = torch.cat((target_overall, target))

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())

    accuracy_overall = accuracy(prediction_overall.type(torch.int64), target_overall.type(torch.int64)).item()
    loss_overall = np.mean(losses)
    print(f'Training\t->\tLoss: {loss_overall:.3f}\tAcc: {accuracy_overall:.3f},')
    return loss_overall, accuracy_overall


def validate_single_epoch(model, dataloader, loss_fn, device):
    model.eval()
    accuracy = Accuracy().to(device)
    prediction_overall = torch.empty((0,)).to(device)
    target_overall = torch.empty((0,)).to(device)
    losses = []
    with torch.no_grad():
        for input, target in dataloader:
            input, target = input.to(device), target.to(device)

            # calculate loss
            prediction = model(input)
            loss = loss_fn(prediction, target)

            _, predicted = torch.max(prediction.data, dim=1)

            prediction_overall = torch.cat((prediction_overall, predicted))
            target_overall = torch.cat((target_overall, target))
            losses.append(loss.item())

    accuracy_overall = accuracy(prediction_overall.type(torch.int64), target_overall.type(torch.int64)).item()
    loss_overall = np.mean(losses)
    print(f'Validation\t->\tLoss: {loss_overall:.3f}\tAcc: {accuracy_overall:.3f},')

    return loss_overall, accuracy_overall


def train(model, train_dataloader, val_dataloader, loss_fn, optimiser, device, epochs):
    train_data = []
    val_data = []
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_data.append(train_single_epoch(model, train_dataloader, loss_fn, optimiser, device))
        val_data.append(validate_single_epoch(model, val_dataloader, loss_fn, device))
        print("---------------------------")
    print("Finished training")

    train_data = np.array(train_data)
    val_data = np.array(val_data)
    training_data = np.concatenate((train_data, val_data), axis=1)
    df_training_data = pd.DataFrame(training_data, columns=['train_loss', 'train_acc', 'val_loss', 'val_acc'])
    df_training_data.index = df_training_data.index+1
    df_training_data.index.name = 'epoch'

    return df_training_data


def save_training_data(df_training_data, path_experiment=None, model=None):
    import os

    fig, axs = plt.subplots(2, 1)
    sns.lineplot(data=df_training_data, x=df_training_data.index, y='train_loss', label='train_loss', ax=axs[0])
    sns.lineplot(data=df_training_data, x=df_training_data.index, y='val_loss', label='val_loss', ax=axs[0])
    axs[0].set_ylabel('Loss')
    sns.lineplot(data=df_training_data, x=df_training_data.index, y='train_acc', label='train_acc', ax=axs[1])
    sns.lineplot(data=df_training_data, x=df_training_data.index, y='val_acc', label='val_acc', ax=axs[1])
    axs[1].set_ylabel('Accuracy')

    if path_experiment:
        try:
            os.mkdir(path_experiment)
        except FileExistsError:
            pass

        print(f'Saving results to: {path_experiment}')
        path_df_training_data = os.path.join(path_experiment, 'df_training_data.csv')
        path_training_plot = os.path.join(path_experiment, 'training_plot.jpg')
        path_model = os.path.join(path_experiment, 'model.pth')

        df_training_data.to_csv(path_df_training_data)
        fig.savefig(path_training_plot)
        torch.save(model.state_dict(), path_model)


if __name__ == "__main__":
    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = GTZANDataset(path_annotation_original,
                       n_samples=None,
                       transformation=mel_spectrogram,
                       target_sample_rate=sample_rate,
                       num_samples=num_samples,
                       device=device)

    train_dataloader, train_dataset = create_data_loader(path_annotation_original,
                                                         n_samples=10 if debug else None,
                                                         transformation=mel_spectrogram,
                                                         target_sample_rate=sample_rate,
                                                         num_samples=num_samples,
                                                         device=device,
                                                         batch_size=batch_size,
                                                         usage='train')

    val_dataloader, val_dataset = create_data_loader(path_annotation_original,
                                                     n_samples=10 if debug else None,
                                                     transformation=mel_spectrogram,
                                                     target_sample_rate=sample_rate,
                                                     num_samples=num_samples,
                                                     device=device,
                                                     batch_size=batch_size,
                                                     usage='train')

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    print(cnn)
    summary(cnn, (1, 1, 400, 400))

    # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=learning_rate)

    # train model
    df_training_data = train(model=cnn,
                          train_dataloader=train_dataloader,
                          val_dataloader=val_dataloader,
                          loss_fn=loss_fn,
                          optimiser=optimiser,
                          device=device,
                          epochs=epochs)

    save_training_data(df_training_data, path_experiment=path_experiment, model=cnn)

