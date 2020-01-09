from torch.utils.data import BatchSampler
import numpy as np
import torch
import random

from matplotlib import pyplot as plt


# class OnlineSampler(BatchSampler):
#     def __init__(self, labels, n_classes, n_samples):
#         # Quick fix for different formats of data
#         # TODO: Refactor
#         if type(labels) == list:
#             self.labels = torch.FloatTensor(labels)
#         else:    
#             self.labels = labels

#         self.n_classes = n_classes
#         self.n_samples = n_samples
#         self.n_dataset = len(self.labels)
#         self.batch_size = self.n_samples * self.n_classes
#         self.labels_set = list(set(self.labels.numpy()))

#         self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
#                                  for label in self.labels_set}
#         for l in self.labels_set:
#             np.random.shuffle(self.label_to_indices[l])
#         self.used_label_indices_count = {label: 0 for label in self.labels_set}

#     def __iter__(self):
#         self.count = 0
#         while self.count + self.batch_size <= self.n_dataset:
#             classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
#             indices = []
#             for class_ in classes:
#                 indices.extend(self.label_to_indices[class_][
#                                self.used_label_indices_count[class_]:self.used_label_indices_count[
#                                                                          class_] + self.n_samples])
#                 self.used_label_indices_count[class_] += self.n_samples
#                 if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
#                     np.random.shuffle(self.label_to_indices[class_])
#                     self.used_label_indices_count[class_] = 0
#             random.shuffle(indices)
#             yield indices
#             self.count += self.batch_size

#     def __len__(self):
#         return self.n_dataset // self.batch_size

def create_pids2idxs(labels):
    """Creates a mapping between pids and indexes of images for that pid.
    Returns:
        2D List with pids => idx
    """
    pid2imgs = {}
    for idx, label in enumerate(labels):
        if label not in pid2imgs:
            pid2imgs[label] = [idx]
        else:
            pid2imgs[label].append(idx)
    return pid2imgs


class OnlineSampler(BatchSampler):
    """Sampler to create batches with P x K.
        
       Only returns indices.
        
    """
    def __init__(self, labels, n_classes=2, n_samples=2, drop_last=True):
        if type(labels) != list:
           self.labels = labels.tolist()
        else:    
            self.labels = labels

        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_classes * self.n_samples
        self.drop_last = drop_last

        self.pid2imgs = create_pids2idxs(self.labels)

    def __iter__(self):
        """Iterator over all images in dataset.
        Picks images accoding to Batch Hard.
        P: #pids in batch
        K: #images per pid
        
        Sorts PIDs randomly and iterates over each pid once.
        Fills batch by selecting K images for each PID. If expected size
        is reach, batch is yielded.
        """

        batch = []
        P_perm = np.random.permutation(len(self.pid2imgs))
        for p in P_perm:
            images = self.pid2imgs[p]
            K_perm = np.random.permutation(len(images))
            # fill up by repeating the permutation
            if len(images) < self.n_samples:
                K_perm = np.tile(K_perm, self.n_samples//len(images))
                left = self.n_samples - len(K_perm)
                K_perm = np.concatenate((K_perm, K_perm[:left]))
            for k in range(self.n_samples):
                batch.append(images[K_perm[k]])
        
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 1 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.labels) // self.n_classes
        else:
            return (len(self.labels) + self.batch_size - 1) // self.n_classes


if __name__ == "__main__":
    from torchvision.datasets import MNIST
    from datasets import Cars3D
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from utils import make_directory
    from BaseData import BaseData
    from torchvision.utils import make_grid

    data_path = make_directory("../datasets/")
    sampling_method = "triplet"
    
    # data_transforms = transforms.Compose([transforms.ToTensor()])
    # train_data = MNIST(root=data_path, train=True, transform=data_transforms)
    train_data = Cars3D(root=data_path, mode="train")
    train_dataset = BaseData(train_data, sampling_method=sampling_method)

    balanced_sampler = OnlineSampler(train_dataset.targets, n_classes=2, n_samples=2)

    dataloader = DataLoader(train_dataset, batch_sampler=balanced_sampler)
    for idx ,data_items in enumerate(dataloader):
        print(data_items["anchor_target"])
        break
    

 

    