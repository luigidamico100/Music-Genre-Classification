from torch import nn
from torchinfo import summary
import torch.nn.functional as F


class Conv2d(nn.Module):

    def __init__(self, input_channels, output_channels, shape=3, pooling=2, dropout=0.1):
        """
        The __init__ function is called when an object is created from the class and it allows the class to initialize
        the attributes of a class. The self parameter refers to the instance of the object. Using self we can access other
        attributes or methods of same object.

        :param self: Access variables that belongs to the class
        :param input_channels: Define the number of channels in the input tensor
        :param output_channels: Define the number of channels in the output of this layer
        :param shape=3: Define the size of the filter
        :param pooling=2: Specify the size of the pooling layer
        :param dropout=0.1: Avoid overfitting
        :return: The self object
        :doc-author: Trelent
        """
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, padding=shape // 2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(pooling)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_data):
        out = self.conv(input_data)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        return out


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout=.1):
        """
        The __init__ function is the constructor for a class. It is called when you create an instance of a class.
        The self parameter refers to the instance of the object itself.

        :param self: Access variables that belongs to the class
        :param in_features: Define the number of input features
        :param out_features: Define the number of nodes in the output layer
        :param dropout=.1: Set the dropout rate of the model
        :return: The object of the class
        :doc-author: Trelent
        """
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_data):
        x = self.linear(input_data)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MyCNNNetwork(nn.Module):

    def __init__(self,
                 num_channels=16,
                 num_classes=10,
                 dropout=0.1,
                 return_embeddings=False):
        super(MyCNNNetwork, self).__init__()

        self.layer1 = Conv2d(1, num_channels, pooling=(2, 3), dropout=dropout)
        self.layer2 = Conv2d(num_channels, num_channels * 2, pooling=(2, 3), dropout=dropout)
        self.layer3 = Conv2d(num_channels * 2, num_channels * 4, pooling=(2, 3), dropout=dropout)
        self.layer4 = Conv2d(num_channels * 4, num_channels * 8, pooling=(2, 3), dropout=dropout)

        self.flatten = nn.Flatten()
        self.linear1 = Linear(num_channels * 8, num_channels * 4, dropout=dropout)
        self.linear2 = nn.Linear(num_channels * 4, num_classes)

        self.return_embeddings = return_embeddings

    def forward(self, input_data):
        x = input_data
        x = x.unsqueeze(1)  # (b, f, t) -> (b, c=1, f, t)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.max_pool2d(x, kernel_size=x.shape[2:])  # (b, c, f, t) -> (b, c, 1, 1)
        # x = x.reshape(len(x), -1)
        x = self.flatten(x)  # (b, c, 1, 1) -> (b, c)
        if self.return_embeddings:
            return x
        x = self.linear1(x)
        x = self.linear2(x)

        return x


class CNNNetwork(nn.Module):

    def __init__(self, return_embeddings=False):
        """
        The __init__ function initializes the class. It is automatically called when you create an object of your class.
        The self parameter refers to the instance of the object that calls this function.
        You can think of it as &quot;this&quot; in C++ or Java, but Python doesn't have these pointers.

        :param self: Access the attributes and methods of the class in python
        :param return_embeddings=False: Tell the model whether or not to return the embeddings
        :return: The input and output dimensions of your model
        :doc-author: Trelent
        """
        # TODO: Insert batch normalization between conv e relu
        super().__init__()

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
        # self.linear = nn.Linear(128, 10)
        self.linear1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(32, 10),
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        # input_data = torch tensor of shape (b, f, t)

        x = input_data

        x = x.unsqueeze(1)  # (b, f, t) -> (b, c=1, f, t)
        x = self.batch_normalization(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = self.flatten(x)  # (b, c=128, f=1, t=1) -> (b, c)
        if self.return_embeddings:
            return x
        x = self.linear1(x)
        x = self.linear2(x)
        logits = self.linear3(x)
        # logits = self.linear(x)
        # predictions = self.softmax(logits)
        return logits


if __name__ == "__main__":
    from genre_classification.models.config import device
    import torch

    cnn = MyCNNNetwork().to(device)
    # print(summary(cnn, (1, 64, 603)))

    print(summary(cnn, (1, 128, 603)))

    a = torch.rand(2, 1000, 1000)
    cnn(a)
