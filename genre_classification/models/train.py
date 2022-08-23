import copy
import torch
from torch import nn
from torchinfo import summary
from torchmetrics import Accuracy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from genre_classification.models.cnn import CNNNetwork, MyCNNNetwork
from genre_classification.models.dataset import create_data_loader
from genre_classification.paths import (
    path_annotations,
    path_class_to_genre_map,
    path_genre_to_class_map,
    get_path_experiment
)
from genre_classification.models import config
import json
from genre_classification.models.config import MyLogger


def train_single_epoch(model, dataloader, loss_fn, optimiser, device):
    """
    The train_single_epoch function trains the model for a single epoch.
    It iterates through all batches in the dataloader and updates weights accordingly.
    The function returns the average loss and accuracy of this epoch.

    :param model: Pass the model that is being trained
    :param dataloader: Iterate over the dataset
    :param loss_fn: Calculate the loss of the model
    :param optimiser: Update the weights of the model
    :param device: Tell torch which device to use
    :return: The overall loss and accuracy of the model on a single epoch
    :doc-author: Trelent
    """
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
    """
    The validate_single_epoch function is used to validate the model on a single epoch.
    It takes as input:
    - model: the model we want to validate (torch.nn.Module)
    - dataloader: a torch DataLoader object containing our validation data (torch.utils.data)
    - loss_fn: the loss function that calculates how far off predictions are from actual values (torch)  # noQA E501
    - device: where we want to run our computations, either GPU or CPU (str). If 'GPU' then .cuda() will be called on the network and inputs; if

    :param model: Specify the model that is used for training
    :param dataloader: Obtain the data
    :param loss_fn: Define the loss function used during training
    :param device: Tell the network whether it should run on the cpu or gpu
    :return: The loss and accuracy of the model on a given dataset
    :doc-author: Trelent
    """
    model.eval()
    accuracy = Accuracy().to(device)
    prediction_overall = torch.empty((0,)).to(device)
    target_overall = torch.empty((0,)).to(device)
    losses = []
    with torch.no_grad():
        for input, target in dataloader:
            input, target = input.to(device), target.to(device)
            b, c, f, t = input.shape  # batchs, chunks, freq., time
            input = input.view(-1, f, t)  # (b, c, f, t) -> (b*c, f, t)
            prediction = model(input)
            prediction = prediction.view(b, c, -1).mean(dim=1)  # Check that
            loss = loss_fn(prediction, target)
            _, predicted = torch.max(prediction.data, dim=1)

            prediction_overall = torch.cat((prediction_overall, predicted))
            target_overall = torch.cat((target_overall, target))
            losses.append(loss.item())

    accuracy_overall = accuracy(prediction_overall.type(torch.int64), target_overall.type(torch.int64)).item()
    loss_overall = np.mean(losses)
    print(f'Validation\t->\tLoss: {loss_overall:.3f}\tAcc: {accuracy_overall:.3f},')

    return loss_overall, accuracy_overall


def train(model, train_dataloader, val_dataloader, loss_fn, optimiser, device, epochs, mylogger):
    """
    The train function trains a model for a given number of epochs.
    It returns the training and validation history as well as the best model found.


    :param model: Specify the model to be trained
    :param train_dataloader: Specify the training data
    :param val_dataloader: Specify the validation data
    :param loss_fn: Specify the loss function used for training
    :param optimiser: Specify the optimiser used for training
    :param device: Tell the model where to run
    :param epochs: Specify the number of epochs to train for
    :param mylogger: Save the training history in a csv file
    :return: Two dataframes:
    :doc-author: Trelent
    """

    train_data = []
    val_data = []
    start_time = time.time()
    best_val_loss = np.inf
    best_epoch = None
    for i in range(1, epochs + 1):
        print(f"Epoch {i}")
        train_data.append(train_single_epoch(model, train_dataloader, loss_fn, optimiser, device))
        val_data.append(validate_single_epoch(model, val_dataloader, loss_fn, device))
        if val_data[-1][0] < best_val_loss:
            best_model = copy.deepcopy(model)
            best_val_loss = val_data[-1][0]
            best_epoch = i
            print('Best model found')
        print("---------------------------")
    end_time = time.time()
    print('Training finished! :)')
    mylogger.write(f'Best model found in epoch: {best_epoch}')
    mylogger.write(f'Elapsed time: {(end_time - start_time) / 60.:.2f} mins')

    train_data = np.array(train_data)
    val_data = np.array(val_data)
    training_data = np.concatenate((train_data, val_data), axis=1)
    df_training_history = pd.DataFrame(training_data, columns=['train_loss', 'train_acc', 'val_loss', 'val_acc'])
    df_training_history.index = df_training_history.index + 1
    df_training_history.index.name = 'epoch'

    return df_training_history, best_model


def save_training_data(df_training_history, params, model, mylogger):
    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 1)
        sns.lineplot(data=df_training_history, x=df_training_history.index, y='train_loss', label='train_loss',
                     ax=axs[0])
        sns.lineplot(data=df_training_history, x=df_training_history.index, y='val_loss', label='val_loss', ax=axs[0])
        axs[0].set_ylabel('Loss')
        sns.lineplot(data=df_training_history, x=df_training_history.index, y='train_acc', label='train_acc', ax=axs[1])
        sns.lineplot(data=df_training_history, x=df_training_history.index, y='val_acc', label='val_acc', ax=axs[1])
        axs[1].set_ylabel('Accuracy')
        plt.tight_layout()

    experiment_name = params['experiment_name']

    path_training_log = get_path_experiment(experiment_name, file_type='training_log')
    path_df_training_history = get_path_experiment(experiment_name, file_type='df_training_history')
    path_training_plot = get_path_experiment(experiment_name, file_type='training_plot')
    path_best_model = get_path_experiment(experiment_name, file_type='best_model')
    path_params = get_path_experiment(experiment_name, file_type='json')

    print(f'Saving training_log to {path_training_log}')
    mylogger.write_on_file(path_training_log)
    print(f'Saving df_training_history to {path_df_training_history}')
    df_training_history.to_csv(path_df_training_history)
    print(f'Saving training_plot to {path_training_plot}')
    fig.savefig(path_training_plot)
    print(f'Saving best_model to {path_best_model}')
    torch.save(model.state_dict(), path_best_model)
    print(f'Saving params json to {path_params}')
    with open(path_params, 'w') as outfile:
        json.dump(params, outfile)


def main(config):
    parsed_params = config.parse_params(config, reason='training')
    mylogger = MyLogger()

    print('----- Parsed params -----')
    mylogger.write(str(parsed_params))
    print()

    # save_params(train_debug_mode, n_examples, experiment_name, epochs, learning_rate, chunks_len_sec)

    mel_spectrogram_params = {'n_fft': parsed_params['melspec_fft'],
                              'hop_length': parsed_params['melspec_hop_length'],
                              'n_mels': parsed_params['melspec_n_mels']}

    train_dataloader, train_dataset = create_data_loader(set_='train',
                                                         batch_size=parsed_params['batch_size'],
                                                         mel_spectrogram_params=mel_spectrogram_params,
                                                         path_annotations_file=path_annotations,
                                                         path_class_to_genre_map=path_class_to_genre_map,
                                                         path_genre_to_class_map=path_genre_to_class_map,
                                                         training=True,
                                                         augment=True,
                                                         n_examples=parsed_params['n_examples'],
                                                         target_sample_rate=config.sample_rate,
                                                         chunks_len_sec=parsed_params['chunks_len_sec'],
                                                         device=config.device, )

    val_dataloader, val_dataset = create_data_loader(set_='val',
                                                     batch_size=parsed_params['batch_size'],
                                                     mel_spectrogram_params=mel_spectrogram_params,
                                                     path_annotations_file=path_annotations,
                                                     path_class_to_genre_map=path_class_to_genre_map,
                                                     path_genre_to_class_map=path_genre_to_class_map,
                                                     training=False,
                                                     augment=False,
                                                     n_examples=parsed_params['n_examples'],
                                                     target_sample_rate=config.sample_rate,
                                                     chunks_len_sec=parsed_params['chunks_len_sec'],
                                                     device=config.device, )

    print('----- Dataset tensor shapes -----')
    print(f'train_dataset\t-> input shape: {train_dataset[0][0].shape}')
    print(f'val_dataset  \t-> input shape: {val_dataset[0][0].shape}')
    print()

    # construct model and assign it to device
    cnn = MyCNNNetwork(dropout=parsed_params['dropout']).to(config.device)
    mylogger.write(str(cnn))
    input_size_example = (parsed_params['batch_size'], train_dataset[0][0].size(0), train_dataset[0][0].size(1))
    mylogger.write(str(summary(cnn, input_size_example)))

    # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=parsed_params['learning_rate'])

    # train model
    df_training_history, best_model = train(model=cnn,
                                            train_dataloader=train_dataloader,
                                            val_dataloader=val_dataloader,
                                            loss_fn=loss_fn,
                                            optimiser=optimiser,
                                            device=config.device,
                                            epochs=parsed_params['epochs'],
                                            mylogger=mylogger)

    save_training_data(df_training_history, params=parsed_params, model=best_model, mylogger=mylogger)


if __name__ == "__main__":
    main(config)
