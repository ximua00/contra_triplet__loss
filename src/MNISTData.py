from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from ContrastiveSampler import ContrastiveSampler
from utils import make_directory


data_path = make_directory("../datasets/")


class MNISTData(Dataset):
    def __init__(self, data, sampler):
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf']
        self.is_train = data.train
        self.sampler = sampler
        self.data = data
        self.data_length = len(data)
        self.n_groundtruths = self.groundtruths_per_class()

    def groundtruths_per_class(self):
        n_groundtruths = dict()
        for class_id, class_idxs in self.sampler.class_idxs.items():
            n_groundtruths[class_id] = len(class_idxs)
        return n_groundtruths

    def __getitem__(self, idx):
        data_items = dict()
        anchor, anchor_target = self.data[idx]
        data_items["anchor"] = anchor
        data_items["anchor_target"] = anchor_target
        if self.is_train:
            duplet_id, is_pos = self.sampler.sample_data(idx, anchor_target)
            duplet, _ = self.data[duplet_id]
            data_items["duplet"] = duplet
            data_items["is_pos"] = is_pos

        return data_items

    def __len__(self):
        return self.data_length

    def show_image(self, idx):
        im = self.data.data[idx]
        trans = transforms.ToPILImage()
        im = trans(im)
        im.show()

    def plot_2D_embeddings(self, embeddings, targets, xlim=None, ylim=None):
        plt.figure(figsize=(10, 10))
        for i in range(10):
            inds = np.where(targets == i)[0]
            plt.scatter(embeddings[inds, 0], embeddings[inds,
                                                        1], alpha=0.5, color=self.colors[i])
        if xlim:
            plt.xlim(xlim[0], xlim[1])
        if ylim:
            plt.ylim(ylim[0], ylim[1])
        plt.legend(self.classes)
        plt.show()


if __name__ == "__main__":
    mnist_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = MNIST(root=data_path, train=True, transform=mnist_transforms)
    train_dataset = MNISTData(train_data)

    dataloader = DataLoader(train_dataset, batch_size=4)
    for id, (anchor, duplet, is_pos) in enumerate(dataloader):
        # print(anchor.size())
        # print(duplet.size())
        # print(is_pos.size())
        # print(id)
        break
