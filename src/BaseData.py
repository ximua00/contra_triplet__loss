import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from samplers import ContrastiveSampler, TripletSampler

from samplers import *


class BaseData(Dataset):
    def __init__(self, data, sampling_method):
        self.data = data
        if sampling_method == "contrastive":
            self.sampler = ContrastiveSampler(self.data)
        elif sampling_method == "triplet":
            self.sampler = TripletSampler(self.data)

        self.is_train = data.train
        self.data_length = len(data)
        self.n_groundtruths = self.groundtruths_per_class()
        self.is_triplet = self.sampler.is_triplet
        self.targets = self.data.targets

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
            if not self.is_triplet:
                data_items["duplet"], data_items["is_pos"] = self.__getitem_duplet(
                    idx, anchor_target)
            else:
                data_items["pos"], data_items["neg"] = self.__getitem_triplet(
                    idx, anchor_target)

        return data_items

    def __getitem_duplet(self, idx, anchor_target):
        duplet_id, is_pos = self.sampler.sample_data(idx, anchor_target)
        duplet, _ = self.data[duplet_id]
        return duplet, is_pos

    def __getitem_triplet(self, idx, anchor_target):
        pos_id, neg_id = self.sampler.sample_data(idx, anchor_target)
        pos, _ = self.data[pos_id]
        neg, _ = self.data[neg_id]
        return pos, neg

    def __len__(self):
        return self.data_length

    def show_image(self, idx):
        im = self.data.data[idx]
        trans = transforms.ToPILImage()
        im = trans(im)
        im.show()


if __name__ == "__main__":
    from utils import make_directory
    from torchvision.datasets import MNIST, CIFAR10
    from networks import CIFAREmbeddingNet
    from networks import SiameseNet, TripletNet
    from datasets import Cars3D, CarsEPFL

    data_path = make_directory("../datasets/")

    net = CIFAREmbeddingNet()
    model = TripletNet(net)
    sampling_method = "triplet"
    # data_transforms = transforms.Compose([transforms.ToTensor()])
    # train_data = MNIST(root=data_path, train=True, transform=data_transforms)
    train_data = CarsEPFL(root=data_path, mode="train")
    query_data = CarsEPFL(root=data_path, mode="query")
    gallery_data = CarsEPFL(root=data_path, mode="gallery")
    
    train_dataset = BaseData(train_data, sampling_method=sampling_method)
    query_dataset = BaseData(query_data, sampling_method=sampling_method)
    gallery_dataset = BaseData(gallery_data, sampling_method=sampling_method)

    print(train_dataset.n_groundtruths)
    print(query_dataset.n_groundtruths)
    print(gallery_dataset.n_groundtruths)
    
    # dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # for data_items in dataloader:
    #     print(data_items.keys())
        
