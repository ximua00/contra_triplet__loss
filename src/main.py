import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch

from BaseData import BaseData
from datasets import *
from losses import *
from networks import *
from OnlineSampler import OnlineSampler

from train import train
from metrics import mean_average_precision
import utils
from config import device


dataset = "MNIST"
sampling_method = "batch_soft"
n_epochs = 50
data_path = utils.make_directory("../datasets/")
batch_size = 64
num_workers = 4
lr = 1e-3
step_size = 8
margin = 1.0
embedding_dim=32
n_classes = 8  
n_samples = 64

experiment_name = dataset + "_" + str(embedding_dim) + "_" + sampling_method
print(experiment_name)

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
    train_data = FashionMNIST(root=data_path, train=True, transform=data_transforms)
    query_data = FashionMNIST(root=data_path, train=False, transform=data_transforms)
    gallery_data = FashionMNIST(root=data_path, train=True, transform=data_transforms)
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
elif dataset == "CarsEPFL":
    embedding_net = CIFAREmbeddingNet(embedding_dim)
    train_data = CarsEPFL(root=data_path, mode="train")
    query_data = CarsEPFL(root=data_path, mode="query")
    gallery_data = CarsEPFL(root=data_path, mode="gallery")
elif dataset == "CarsShapeNet":
    embedding_net = CIFAREmbeddingNet(embedding_dim)
    train_data = CarsShapeNet(root=data_path, mode="train")
    query_data = CarsShapeNet(root=data_path, mode="query")
    gallery_data = CarsShapeNet(root=data_path, mode="gallery")

if sampling_method == "contrastive":
    criterion = ContrastiveLoss(margin=margin)
    model = SiameseNet(embedding_net).to(device)
elif sampling_method == "triplet":
    criterion = TripletLoss(margin=margin)
    model = TripletNet(embedding_net).to(device)
elif sampling_method == "batch_hard":
    criterion = BatchSoft(margin=margin,T=0.00001) # very small temperature
    model = embedding_net.to(device)
elif sampling_method == "batch_soft":
    criterion = BatchSoft(margin=margin)
    model = embedding_net.to(device)
else:
    raise NotImplementedError


train_dataset = BaseData(train_data, sampling_method)
query_dataset = BaseData(query_data, sampling_method)
gallery_dataset = BaseData(gallery_data, sampling_method)

if sampling_method == "batch_hard" or "batch_soft":
    balanced_sampler = OnlineSampler(train_dataset.targets, n_classes=n_classes, n_samples=n_samples)
    train_loader = DataLoader(train_dataset, batch_sampler=balanced_sampler)
else:
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

query_loader = DataLoader(
    query_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
gallery_loader = DataLoader(
    gallery_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size)


train(model, criterion, train_loader, query_loader, gallery_loader, optimizer, scheduler, experiment_name, sampling_method, n_epochs=n_epochs)

model = utils.load_model(model, experiment_name)
print(experiment_name)
mAP = mean_average_precision(model, query_loader, gallery_loader,k=1)
print("k=1",mAP)
mAP = mean_average_precision(model, query_loader, gallery_loader,k=50)
print("k=50",mAP)
mAP = mean_average_precision(model, query_loader, gallery_loader,k=100)
print("k=100",mAP)
mAP = mean_average_precision(model, query_loader, gallery_loader)
print(mAP)


# embeddings_matrix, targets_vector = utils.get_dataset_embeddings(
#     model, train_dataloader)
# colors, classes = utils.get_colorcode(dataset)
# utils.plot_embeddings(embeddings_matrix, targets_vector, colors, classes)
