import torch
import torchaudio
from torch import nn
from torchinfo import summary
from torchmetrics import Accuracy
import numpy as np
import pickle
from genre_classification.models.cnn import CNNNetwork
from genre_classification.models.dataset import create_data_loader, GTZANDataset
from genre_classification.paths import path_annotation_original, path_model, path_training_data
from genre_classification.models.config import (
    device,
    batch_size,
    epochs,
    learning_rate,
    sample_rate,
    num_samples,
)


def train_single_epoch(model, dataloader, loss_fn, optimiser, device):
    model.train()
    accuracy = Accuracy()
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

    accuracy_overall = accuracy(prediction_overall.type(torch.int64), target_overall.type(torch.int64))
    loss_overall = np.mean(losses)
    print(f'Training\t->\tLoss: {loss_overall:.3f}\tAcc: {accuracy_overall.item():.3f},')
    return loss_overall, accuracy_overall


def validate_single_epoch(model, dataloader, loss_fn, device):
    model.eval()
    accuracy = Accuracy()
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

    accuracy_overall = accuracy(prediction_overall.type(torch.int64), target_overall.type(torch.int64))
    loss_overall = np.mean(losses)
    print(f'Validation\t->\tLoss: {loss_overall:.3f}\tAcc: {accuracy_overall.item():.3f},')

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
    return train_data, val_data


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
                                                         n_samples=None,
                                                         transformation=mel_spectrogram,
                                                         target_sample_rate=sample_rate,
                                                         num_samples=num_samples,
                                                         device=device,
                                                         batch_size=batch_size,
                                                         usage='train')

    val_dataloader, val_dataset = create_data_loader(path_annotation_original,
                                                     n_samples=None,
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
    training_data = train(model=cnn,
                          train_dataloader=train_dataloader,
                          val_dataloader=val_dataloader,
                          loss_fn=loss_fn,
                          optimiser=optimiser,
                          device=device,
                          epochs=epochs)

    # save model
    print(f"Saving model to {path_model}")
    torch.save(cnn.state_dict(), path_model)
    with open(path_training_data, 'wb') as f:
        pickle.dump(f, training_data)
