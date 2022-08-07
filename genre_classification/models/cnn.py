from torch import nn
from torchinfo import summary
import torch.nn.functional as F


class CNNNetwork(nn.Module):

    def __init__(self, print_forward_tensors_shape=False):
        # TODO: Insert batch normalization between conv e relu
        super().__init__()
        self.print_forward_tensors_shape = print_forward_tensors_shape

        self.batch_normalization = nn.BatchNorm2d(1)
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        '''
        input_data = torch tensor of shape (b, f, t)

        '''
        x = input_data
        if self.print_forward_tensors_shape:
            print(x.shape)
        x = x.unsqueeze(1) # (b, f, t) -> (b, c=1, f, t)
        if self.print_forward_tensors_shape:
            print(x.shape)
        x = self.batch_normalization(x)
        if self.print_forward_tensors_shape:
            print(x.shape)
        x = self.conv1(x)
        if self.print_forward_tensors_shape:
            print(x.shape)
        x = self.conv2(x)
        if self.print_forward_tensors_shape:
            print(x.shape)
        x = self.conv3(x)
        if self.print_forward_tensors_shape:
            print(x.shape)
        x = self.conv4(x)
        if self.print_forward_tensors_shape:
            print(x.shape)
        # x = self.conv5(x)
        # if self.print_forward_tensors_shape:
        #     print(x.shape)
        #     print('--- end of conv layers ---')
        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        if self.print_forward_tensors_shape:
            print(x.shape)
        x = self.flatten(x)     # (b, c=128, f=1, t=1) -> (b, c)
        if self.print_forward_tensors_shape:
            print(x.shape)
        logits = self.linear(x)
        if self.print_forward_tensors_shape:
            print(logits.shape)
        predictions = self.softmax(logits)
        if self.print_forward_tensors_shape:
            print(predictions.shape)
        return predictions


class CNNNetwork_original(nn.Module):

    def __init__(self, print_forward_tensors_shape=False):
        super().__init__()
        self.print_forward_tensors_shape = print_forward_tensors_shape

        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        if self.print_forward_tensors_shape:
            print(x.shape)
        x = self.conv2(x)
        if self.print_forward_tensors_shape:
            print(x.shape)
        x = self.conv3(x)
        if self.print_forward_tensors_shape:
            print(x.shape)
        x = self.conv4(x)
        if self.print_forward_tensors_shape:
            print(x.shape)
        x = self.flatten(x)
        if self.print_forward_tensors_shape:
            print(x.shape)
        logits = self.linear(x)
        if self.print_forward_tensors_shape:
            print(logits.shape)
        predictions = self.softmax(logits)
        if self.print_forward_tensors_shape:
            print(predictions.shape)
        return predictions


if __name__ == "__main__":
    from genre_classification.models.config import device
    cnn = CNNNetwork(print_forward_tensors_shape=True).to(device)
    summary(cnn, (1, 1, 64, 44))

    import torch
    input_tensor = torch.rand(1, 1, 20, 20)
    # if device=='cuda':
    #     summary(cnn.cuda(), (1, 64, 44))
    # else:
    #     summary(cnn, (1, 64, 44))

