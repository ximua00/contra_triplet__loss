import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def make_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def save_model(model, experiment_name):
    models_path = make_directory("../models/")
    torch.save(model.state_dict(), os.path.join(
        models_path, experiment_name+'.pt'))


def load_model(model, experiment_name):
    model.load_state_dict(torch.load(
        os.path.join("../models/", experiment_name+'.pt')))
    return model

def send_to_device(data_items, device):
    for key in data_items.keys():
        data_items[key] = data_items[key].to(device)
    return data_items

@torch.no_grad()
def get_dataset_embeddings(model, dataloader):
    model.eval()
    embeddings_matrix = np.zeros(
        (len(dataloader.dataset), model.embedding_dim))
    targets_vector = np.zeros((len(dataloader.dataset)))
    k = 0
    for idx, data_items in enumerate(dataloader):
        n_datapoints = data_items["anchor_target"].size()[0]
        anchor_embeddings = model.get_embedding(
            data_items["anchor"].to(device))

        embeddings_matrix[k:k+n_datapoints,
                          :] = anchor_embeddings.cpu().numpy()
        targets_vector[k:k+n_datapoints] = data_items["anchor_target"].numpy()
        k += n_datapoints

    return embeddings_matrix, targets_vector


def get_colorcode(dataset):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    if dataset == "MNIST":
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    elif dataset == "FashionMNIST":
        classes = ["T-shirt", "Trouser", "Pullover", "Dress",
                   "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle"]
    elif dataset == "CIFAR10":
        classes = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return colors, classes


def plot_2D_embeddings(embeddings, targets, colors, classes, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    for i in range(10):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds,
                                                    1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    plt.show()


def plot_ND_embeddings(embeddings, targets, colors, classes, xlim=None, ylim=None):
    X_embedded = TSNE(n_components=2).fit_transform(embeddings)
    plot_2D_embeddings(X_embedded, targets, colors, classes, xlim=None, ylim=None)


def plot_embeddings(embeddings, targets, colors, classes, xlim=None, ylim=None):
    if embeddings.shape[1] > 2:
        plot_ND_embeddings(embeddings, targets, colors, classes, xlim, ylim)
    else:
        plot_2D_embeddings(embeddings, targets, colors, classes, xlim, ylim)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default="Cars3D",
                        help="dataset to process")
    parser.add_argument('-s', '--sampling_method', default="triplet",
                        help="sampling method")
    parser.add_argument('-e', '--n_epochs', type=int, default=50,
                        help="define the number of random integers")
    parser.add_argument('--data_path', default="../../datasets/",
                        help="root data folder")
    parser.add_argument('-m', '--model', default="lenet",
                        help="network")
    parser.add_argument('-b', '--batch_size', type=int, default=256,
                        help="batch size")
    parser.add_argument('-l', '--lr', type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument('-ss', '--step_size', type=int, default=8,
                        help="scheduler steps")
    parser.add_argument('-m', '--margin', type=float, default=1.0,
                        help="margin")
    parser.add_argument('-emb', '--embedding_dim', type=int, default=32,
                        help="embedding size")
    parser.add_argument('-w', '--num_workers', type=int, default=8,
                        help="number of workers")                        
    parser.add_argument('-b_k', '--n_classes', type=int, default=8,
                        help="number of classes for online sampler")
    parser.add_argument('-b_n', '--n_samples', type=int, default=32,
                        help="number of samples per class for online sampler")
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        help="GPU ID")

    args = parser.parse_args()
    return args

args = config()
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")