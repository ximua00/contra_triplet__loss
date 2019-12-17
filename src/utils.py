import os
import torch
import numpy as np


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


@torch.no_grad()
def get_dataset_embeddings(model, dataloader):
    model.eval()
    embeddings_matrix = np.zeros(
        (len(dataloader.dataset), model.embedding_dim))
    targets_vector = np.zeros((len(dataloader.dataset)))
    k = 0
    for idx, data_items in enumerate(dataloader):
        n_datapoints = data_items["anchor_target"].size()[0]
        anchor_embeddings = model.get_embedding(data_items["anchor"])

        embeddings_matrix[k:k+n_datapoints, :] = anchor_embeddings.numpy()
        targets_vector[k:k+n_datapoints] = data_items["anchor_target"].numpy()
        k += n_datapoints

    return embeddings_matrix, targets_vector
