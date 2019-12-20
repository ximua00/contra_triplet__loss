import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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

        self.model = models.googlenet(pretrained=False, aux_logits=False)
        self.model.fc.out_features = self.embedding_dim

    def forward(self,x):
        return self.model(x)


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