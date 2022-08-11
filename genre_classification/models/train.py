import copy
import torch
from torch import nn
from torchinfo import summary
from torchmetrics import Accuracy
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import pandas as pd
import os
import time
from genre_classification.models.cnn import CNNNetwork
from genre_classification.models.dataset import create_data_loader
from genre_classification.paths import (
    path_annotation_original, 
    path_class_to_genre_map,
    path_genre_to_class_map,
    experiment_name, 
    get_path_experiment
    )
from genre_classification.models.config import (
    device,
    batch_size,
    epochs,
    learning_rate,
    sample_rate,
    chunks_len_sec,
    train_debug_mode,
)
import json
from genre_classification.models.config import MyLogger


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
            b, c, f, t = input.shape  # batchs, chunks, freq., time
            input = input.view(-1, f, t)    # (b, c, f, t) -> (b*c, f, t)
            prediction = model(input)
            prediction = prediction.view(b, c, -1).mean(dim=1) # Check that
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
    train_data = []
    val_data = []
    start_time = time.time()
    best_val_loss = np.inf
    best_epoch = None
    for i in range(1, epochs+1):
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
        sns.lineplot(data=df_training_history, x=df_training_history.index, y='train_loss', label='train_loss', ax=axs[0])
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
    

def get_params(train_debug_mode, experiment_name, epochs, learning_rate, chunks_len_sec):
    parser = argparse.ArgumentParser(description='Training process')
    parser.add_argument('--epochs', type=int, help='epochs number', default=epochs)
    # parser.add_argument('--train_debug_mode', type=bool, help='Train debug mode', default=train_debug_mode, action=argparse.BooleanOptionalAction)
    parser.add_argument('--train_debug_mode', type=str, help='Train debug mode', default=train_debug_mode)
    parser.add_argument('--learning_rate', type=float, help='training learning rate', default=learning_rate)
    parser.add_argument('--experiment_name', type=str, help='experiment name', default=experiment_name)
    parser.add_argument('--chunks_len_sec', type=float, help='Chunks length (sec)', default=chunks_len_sec)
    args = parser.parse_args()
    train_debug_mode = args.train_debug_mode == 'True'
    learning_rate = args.learning_rate
    experiment_name = args.experiment_name
    chunks_len_sec = args.chunks_len_sec
    n_examples = 'all' if not train_debug_mode else 50
    epochs = args.epochs if not train_debug_mode else 3
    
    # mylogger.write(f'train_debug_mode={train_debug_mode}')
    # mylogger.write(f'n_examples={n_examples}')
    # mylogger.write(f'experiment_name={experiment_name}')
    # mylogger.write('')
    # mylogger.write(f'epochs={epochs}')
    # mylogger.write(f'learning_rate={learning_rate}')
    # mylogger.write(f'chunks_len_sec={chunks_len_sec}')
    # mylogger.write('')
    
    params = {}
    params['train_debug_mode'] = train_debug_mode
    params['n_examples'] = n_examples
    params['experiment_name'] = experiment_name
    params['epochs'] = epochs
    params['learning_rate'] = learning_rate
    params['chunks_len_sec'] = chunks_len_sec
    
    return params
    
    
    
#%%

def main(train_debug_mode, experiment_name, epochs, learning_rate, chunks_len_sec):
    
    params = get_params(train_debug_mode, experiment_name, epochs, learning_rate, chunks_len_sec)
    train_debug_mode = params['train_debug_mode']
    n_examples = params['n_examples']
    experiment_name = params['experiment_name']
    epochs = params['epochs']
    learning_rate = params['learning_rate']
    chunks_len_sec = params['chunks_len_sec']
    
    mylogger = MyLogger()
    mylogger.write(str(params))
         
    #save_params(train_debug_mode, n_examples, experiment_name, epochs, learning_rate, chunks_len_sec)

    train_dataloader, train_dataset = create_data_loader(split='train',
                                                     batch_size=batch_size,
                                                     path_annotations_file=path_annotation_original,
                                                     path_class_to_genre_map=path_class_to_genre_map,
                                                     path_genre_to_class_map=path_genre_to_class_map,
                                                     n_examples=n_examples,
                                                     target_sample_rate=sample_rate,
                                                     chunks_len_sec=chunks_len_sec,
                                                     device=device,)

    val_dataloader, val_dataset = create_data_loader(split='val',
                                                     batch_size=batch_size,
                                                     path_annotations_file=path_annotation_original,
                                                     path_class_to_genre_map=path_class_to_genre_map,
                                                     path_genre_to_class_map=path_genre_to_class_map,
                                                     n_examples=n_examples,
                                                     target_sample_rate=sample_rate,
                                                     chunks_len_sec=chunks_len_sec,
                                                     device=device,)

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    mylogger.write(cnn.__str__())
    summary(cnn, (1, 64, 603))

    # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=learning_rate)

    # train model
    df_training_history, best_model = train(model=cnn,
                                         train_dataloader=train_dataloader,
                                         val_dataloader=val_dataloader,
                                         loss_fn=loss_fn,
                                         optimiser=optimiser,
                                         device=device,
                                         epochs=epochs,
                                         mylogger=mylogger)

    save_training_data(df_training_history, params=params, model=best_model, mylogger=mylogger)


if __name__ == "__main__":
    main(train_debug_mode, experiment_name, epochs, learning_rate, chunks_len_sec)
