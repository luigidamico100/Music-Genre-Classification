from torch import nn
from torchinfo import summary
import torch.nn.functional as F



class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, pooling=2, dropout=0.1):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, padding=shape//2)
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
    def __init__(self, in_features, out_features, dropout=0.1, relu_layer=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.relu_layer = relu_layer
        
    def forward(self, input_data):
        x = self.linear(input_data)
        x = self.bn(x)
        if self.relu_layer:
            x = self.relu(x)
        return x
        
        
    
    
class CNNNetwork_my(nn.Module):
    
    def __init__(self, num_channels=16, 
                       num_classes=10):
    
        super(CNNNetwork_my, self).__init__()
        
        self.layer1 = Conv_2d(1, num_channels, pooling=(2, 3))
        self.layer2 = Conv_2d(num_channels, num_channels * 2, pooling=(2, 3))
        self.layer3 = Conv_2d(num_channels * 2, num_channels * 4, pooling=(2, 3))
        self.layer4 = Conv_2d(num_channels * 4, num_channels * 8, pooling=(2, 3))
        
        self.flatten = nn.Flatten()
        self.linear1 = Linear(num_channels*8, num_channels*4)
        self.linear2 = Linear(num_channels*4, num_channels*2)
        self.linear3 = Linear(num_channels*2, num_classes, relu_layer=False)
        
        
    def forward(self, input_data):
        x = input_data
        x = x.unsqueeze(1) # (b, f, t) -> (b, c=1, f, t)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # (b, c, f, t) -> (b, c, 1, 1)
        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        # (b, c, 1, 1) -> (b, c)
        # x = x.reshape(len(x), -1)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        
        return x
    
        
class CNN_tutorial(nn.Module):
    def __init__(self, num_channels=16, 
                       sample_rate=22050, 
                       n_fft=1024, 
                       f_min=0.0, 
                       f_max=11025.0, 
                       num_mels=128, 
                       num_classes=10):
        super(CNN_tutorial, self).__init__()

        self.input_bn = nn.BatchNorm2d(1)

        # convolutional layers
        self.layer1 = Conv_2d(1, num_channels, pooling=(2, 3))
        self.layer2 = Conv_2d(num_channels, num_channels, pooling=(3, 4))
        self.layer3 = Conv_2d(num_channels, num_channels * 2, pooling=(2, 5))
        self.layer4 = Conv_2d(num_channels * 2, num_channels * 2, pooling=(3, 3))
        self.layer5 = Conv_2d(num_channels * 2, num_channels * 4, pooling=(3, 4))

        # dense layers
        self.dense1 = nn.Linear(num_channels * 4, num_channels * 4)
        self.dense_bn = nn.BatchNorm1d(num_channels * 4)
        self.dense2 = nn.Linear(num_channels * 4, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, input_data):
        
        x = input_data

        # input batch normalization
        x = x.unsqueeze(1)
        x = self.input_bn(x)

        # convolutional layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        # reshape. (batch_size, num_channels, 1, 1) -> (batch_size, num_channels)
        x = x.reshape(len(x), -1)

        # dense layers
        x = self.dense1(x)
        x = self.dense_bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x
    



class CNNNetwork(nn.Module):

    def __init__(self, return_embeddings=False):
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
        if self.return_embeddings:
            return x
        x = self.linear1(x)
        x = self.linear2(x)
        logits = self.linear3(x)
        # logits = self.linear(x)
        #predictions = self.softmax(logits)
        return logits


if __name__ == "__main__":
    from genre_classification.models.config import device
    import torch
    
    cnn = CNNNetwork_my().to(device)
    # print(summary(cnn, (1, 64, 603)))

    print(summary(cnn, (1, 128, 603)))

    a = torch.rand(2, 1000, 1000)
    cnn(a)
