from torch import nn
from torchinfo import summary
import torch.nn.functional as F


class CNNNetwork(nn.Module):

    def __init__(self, return_embeddings=False, print_forward_tensors_shape=False):
        # TODO: Insert batch normalization between conv e relu
        super().__init__()
        self.print_forward_tensors_shape = print_forward_tensors_shape
        
        self.return_embeddings = return_embeddings

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
            # nn.Dropout(.1),
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
            # nn.Dropout(.1),
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
            # nn.Dropout(.1),
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
            # nn.Dropout(.1),
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
        # self.linear1 = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.ReLU()
        #     )
        # self.linear2 = nn.Sequential(
        #     nn.Linear(64, 32),
        #     nn.ReLU()
        #     )
        # self.linear3 = nn.Sequential(
        #     nn.Linear(32, 10),
        #     )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        '''
        input_data = torch tensor of shape (b, f, t)

        '''
        x = input_data

        x = x.unsqueeze(1) # (b, f, t) -> (b, c=1, f, t)
        x = self.batch_normalization(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = self.flatten(x)     # (b, c=128, f=1, t=1) -> (b, c)
        if self.print_forward_tensors_shape:
            print(x.shape)
        if self.return_embeddings:
            return x
        # x = self.linear1(x)
        # x = self.linear2(x)
        #logits = self.linear3(x)
        logits = self.linear(x)
        #predictions = self.softmax(logits)
        return logits


if __name__ == "__main__":
    from genre_classification.models.config import device
    cnn = CNNNetwork(print_forward_tensors_shape=True, return_embeddings=True).to(device)
    summary(cnn, (1, 64, 44))

    import torch
    input_tensor = torch.rand(1, 1, 20, 20)
    summary(cnn, (1, 400, 400))

