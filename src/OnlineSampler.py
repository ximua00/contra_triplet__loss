from torch.utils.data import BatchSampler
import numpy as np
import torch


class OnlineSampler(BatchSampler):
    def __init__(self, labels, n_classes, n_samples):
        # Quick fix for different formats of data
        # TODO: Refactor
        if type(labels) == list:
            self.labels = torch.FloatTensor(labels)
        else:    
            self.labels = labels

        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes
        self.labels_set = list(set(self.labels.numpy()))

        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size


if __name__ == "__main__":
    from torchvision.datasets import MNIST
    from datasets import Cars3D
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from utils import make_directory
    from BaseData import BaseData
    data_path = make_directory("../datasets/")
    sampling_method = "triplet"
    
    # data_transforms = transforms.Compose([transforms.ToTensor()])
    # train_data = MNIST(root=data_path, train=True, transform=data_transforms)
    train_data = Cars3D(root=data_path, mode="train")
    train_dataset = BaseData(train_data, sampling_method=sampling_method)

    balanced_sampler = OnlineSampler(train_dataset.targets, n_classes=2, n_samples=2)

    dataloader = DataLoader(train_dataset, batch_sampler=balanced_sampler)
    for e in range(5):
        for idx ,data_items in enumerate(dataloader):
            if idx == 0:
                print(data_items.keys())
    


    