import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch

from ContrastiveSampler import ContrastiveSampler
from MNISTData import MNISTData
from ContrastiveLoss import ContrastiveLoss
from networks import EmbeddingNet, SiameseNet

from train import train
from metrics import mean_average_precision
import utils


data_path = utils.make_directory("../datasets/")
experiment_name = "test"
batch_size = 32
num_workers = 2
lr = 1e-3
step_size = 8
margin = 1.0

mnist_transforms = transforms.Compose([transforms.ToTensor()])
train_data = MNIST(root=data_path, train=True, transform=mnist_transforms)
test_data = MNIST(root=data_path, train=False, transform=mnist_transforms)

#############################################
# DEBUG
test_data.data = test_data.data[:100, :, :]
#############################################

sampler = ContrastiveSampler(train_data)
train_dataset = MNISTData(train_data, sampler)
test_dataset = MNISTData(test_data, sampler)

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=num_workers)

embedding_net = EmbeddingNet()
model = SiameseNet(embedding_net)

criterion = ContrastiveLoss(margin=margin)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size)

# train(model, criterion, train_dataloader, test_dataloader, optimizer, scheduler, experiment_name, n_epochs=20)
model = utils.load_model(model, experiment_name)
embeddings_matrix, targets_vector = utils.get_dataset_embeddings(
    model, train_dataloader)
train_dataset.plot_2D_embeddings(embeddings_matrix, targets_vector)
# mAP = mean_average_precision(model, test_dataloader, train_dataloader, k=10)
# print(mAP)
