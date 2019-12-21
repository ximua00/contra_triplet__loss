import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict


class MNISTEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=2):
        super(MNISTEmbeddingNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm1d(256)


        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5),
                                     self.batchnorm1,
                                     nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5),
                                     self.batchnorm2,
                                     nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2)
                                     )

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                self.batchnorm3,
                                nn.PReLU(),
                                nn.Linear(256, 64),
                                nn.PReLU(),
                                nn.Linear(64, self.embedding_dim)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class CIFAREmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=2):
        super(CIFAREmbeddingNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm1d(256)

 
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5),
                                    self.batchnorm1,
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, stride=2),
                                    nn.Conv2d(32, 64, 5),
                                    self.batchnorm2,
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, stride=2),
                                    nn.Conv2d(64, 128, 2),
                                    self.batchnorm3,
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, stride=2)
                                    )

        self.fc = nn.Sequential(nn.Linear(512, 256),
                                    self.batchnorm4,
                                    nn.ReLU(),
                                    nn.Linear(256, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.embedding_dim)
                                    )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.embedding_dim = embedding_net.embedding_dim

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


if __name__ == "__main__":
    x=torch.rand((1,3,32,32))
    embedding_net = CIFAREmbeddingNet()
    print(embedding_net(x).size())