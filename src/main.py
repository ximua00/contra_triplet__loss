import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch

from samplers import *
from BaseData import BaseData
from datasets import Cars3D
from losses import *
from networks import *

from train import train
from metrics import mean_average_precision
import utils
from config import device


dataset = "Cars3D"
sampling_method = "contrastive"
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
    query_data = MNIST(root=data_path, train=False, transform=data_transforms)
    gallery_data = MNIST(root=data_path, train=True, transform=data_transforms)
elif dataset == "FashionMNIST":
    embedding_net = MNISTEmbeddingNet()
    mean, std = 0.28604059698879553, 0.35302424451492237
    data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
    train_data = FashionMNIST(root=data_path, train=True, transform=data_transforms, download=True)
    query_data = FashionMNIST(root=data_path, train=False, transform=data_transforms, download=True)
    gallery_data = FashionMNIST(root=data_path, train=True, transform=data_transforms, download=True)
elif dataset == "CIFAR10":
    embedding_net = CIFAREmbeddingNet(embedding_dim)
    data_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = CIFAR10(root=data_path, train=True, transform=data_transforms)
    query_data = CIFAR10(root=data_path, train=False, transform=data_transforms)
    gallery_data = CIFAR10(root=data_path, train=True, transform=data_transforms)
elif dataset == "Cars3D":
    embedding_net = CIFAREmbeddingNet(embedding_dim)
    train_data = Cars3D(root=data_path, mode="train")
    query_data = Cars3D(root=data_path, mode="query")
    gallery_data = Cars3D(root=data_path, mode="gallery")


if sampling_method == "contrastive":
    sampler = ContrastiveSampler(train_data)
    criterion = ContrastiveLoss(margin=margin)
    model = SiameseNet(embedding_net).to(device)
elif sampling_method == "triplet":
    sampler = TripletSampler(train_data)
    criterion = TripletLoss(margin=margin)
    model = TripletNet(embedding_net).to(device)


train_dataset = BaseData(train_data, sampler)
query_dataset = BaseData(query_data, sampler)
gallery_dataset = BaseData(gallery_data, sampler)


train_loader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers)
query_loader = DataLoader(
    query_dataset, batch_size=batch_size, num_workers=num_workers)
gallery_loader = DataLoader(
    gallery_dataset, batch_size=batch_size, num_workers=num_workers)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size)


train(model, criterion, train_loader, query_loader, gallery_loader, optimizer, scheduler, experiment_name, n_epochs=n_epochs)

# model = utils.load_model(model, experiment_name)
# mAP = mean_average_precision(model, query_loader, gallery_loader,k=1)
# print("k=1",mAP)
# mAP = mean_average_precision(model, query_loader, gallery_loader,k=50)
# print("k=50",mAP)
# mAP = mean_average_precision(model, query_loader, gallery_loader,k=100)
# print("k=100",mAP)
# mAP = mean_average_precision(model, query_loader, gallery_loader)
# print(mAP)


# embeddings_matrix, targets_vector = utils.get_dataset_embeddings(
#     model, train_dataloader)
# colors, classes = utils.get_colorcode(dataset)
# utils.plot_embeddings(embeddings_matrix, targets_vector, colors, classes)
