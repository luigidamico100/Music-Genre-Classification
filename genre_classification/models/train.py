import torch
import torchaudio
from torch import nn
from torchinfo import summary
from torchmetrics import Accuracy
from genre_classification.models.cnn import CNNNetwork
from genre_classification.models.dataset import create_data_loader, GTZANDataset
from genre_classification.paths import path_annotation_original, path_model
from genre_classification.models.config import (
    device,
    batch_size,
    epochs,
    learning_rate,
    sample_rate,
    num_samples,
)


def train_single_epoch(model, dataloader, loss_fn, optimiser, device):
    for input, target in dataloader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")

def validate_single_epoch(model, dataloader, loss_fn, device):
    accuracy = Accuracy()
    prediction_overall = torch.empty((0,))
    target_overall = torch.empty((0,))
    for input, target in dataloader:
        input, target = input.to(device), target.to(device)
        model.eval()

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        _, predicted = torch.max(prediction.data, dim=1)

        prediction_overall = torch.cat((prediction_overall, predicted))
        target_overall = torch.cat((target_overall, target))

    accuracy_overall = accuracy(prediction_overall.type(torch.int64), target_overall.type(torch.int64))
    print(f'Validation\t->\tLoss: {loss.item()}\tAcc: {accuracy_overall.item()},')

def train(model, train_dataloader, val_dataloader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, train_dataloader, loss_fn, optimiser, device)
        validate_single_epoch(model, val_dataloader, loss_fn, device)
        print("---------------------------")
    print("Finished training")


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

    train_dataloader = create_data_loader(usd, batch_size)

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    print(cnn)
    summary(cnn, (1, 1, 400, 400))

    # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=learning_rate)

    # train model
    train(model=cnn,
          train_dataloader=train_dataloader,
          val_dataloader=train_dataloader,
          loss_fn=loss_fn,
          optimiser=optimiser,
          device=device,
          epochs=epochs)

    # save model
    torch.save(cnn.state_dict(), path_model)
    print(f"Saving model to {path_model}")
