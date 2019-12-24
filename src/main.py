import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch

from samplers import *
from BaseData import BaseData
from losses import *
from networks import *

from train import train
from metrics import mean_average_precision
import utils
from config import device


dataset = "MNIST"
sampling_method = "triplet"
n_epochs = 20
data_path = utils.make_directory("../datasets/")
batch_size = 32
num_workers = 4
lr = 1e-3
step_size = 8
margin = 1.0
embedding_dim=32

experiment_name = dataset + "_" + str(embedding_dim) + "_" + sampling_method

if dataset == "MNIST":
    embedding_net = MNISTEmbeddingNet()
    mean, std = 0.1307, 0.3081
    data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
    train_data = MNIST(root=data_path, train=True, transform=data_transforms)
    test_data = MNIST(root=data_path, train=False, transform=data_transforms)

if dataset == "FashionMNIST":
    embedding_net = MNISTEmbeddingNet()
    mean, std = 0.28604059698879553, 0.35302424451492237
    data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
    train_data = FashionMNIST(root=data_path, train=True, transform=data_transforms, download=True)
    test_data = FashionMNIST(root=data_path, train=False, transform=data_transforms, download=True)

elif dataset == "CIFAR10":
    embedding_net = CIFAREmbeddingNet(embedding_dim)
    data_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = CIFAR10(root=data_path, train=True, transform=data_transforms)
    test_data = CIFAR10(root=data_path, train=False, transform=data_transforms)


if sampling_method == "contrastive":
    sampler = ContrastiveSampler(train_data)
    criterion = ContrastiveLoss(margin=margin)
    model = SiameseNet(embedding_net).to(device)
elif sampling_method == "triplet":
    sampler = TripletSampler(train_data)
    criterion = TripletLoss(margin=margin)
    model = TripletNet(embedding_net).to(device)


train_dataset = BaseData(train_data, sampler)
test_dataset = BaseData(test_data, sampler)

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=num_workers)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size)


train(model, criterion, train_dataloader, test_dataloader, optimizer, scheduler, experiment_name, n_epochs=n_epochs)

# model = utils.load_model(model, experiment_name)
# mAP = mean_average_precision(model, test_dataloader, train_dataloader,k=1)
# print("k=1",mAP)
# mAP = mean_average_precision(model, test_dataloader, train_dataloader,k=50)
# print("k=50",mAP)
# mAP = mean_average_precision(model, test_dataloader, train_dataloader,k=100)
# print("k=100",mAP)
# mAP = mean_average_precision(model, test_dataloader, train_dataloader)
# print(mAP)


# embeddings_matrix, targets_vector = utils.get_dataset_embeddings(
#     model, train_dataloader)
# colors, classes = utils.get_colorcode(dataset)
# utils.plot_embeddings(embeddings_matrix, targets_vector, colors, classes)
