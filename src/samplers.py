from collections import defaultdict
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import random

from utils import make_directory


class MetricSampler:
    def __init__(self, train_data):
        self.train_data = train_data
        self.class_idxs, self.classes_list = self.__get_class_idxs()

    def __get_class_idxs(self):
        class_idxs = defaultdict(list)
        for idx, target in enumerate(self.train_data.targets):
            if type(target) is int:
                class_idxs[target].append(idx)
            else:
                class_idxs[target.item()].append(idx)
        return class_idxs, list(class_idxs.keys())

    def sample_data(self):
        raise NotImplementedError


class ContrastiveSampler(MetricSampler):
    def __init__(self, train_data, pos_threshold=0.25):
        super().__init__(train_data)
        self.pos_threshold = pos_threshold

    @property
    def is_triplet(self):
        return False

    def sample_data(self, anchor_id, anchor_target):
        # flip a coin to decide if positive or negative pair
        is_pos = self.which_class()
        if is_pos == 1:
            pair_id = random.sample(self.class_idxs[anchor_target], k=1)[0]
        else:
            # randomly select another class
            neg_class = random.choice(
                [x for x in self.classes_list if x != anchor_target])
            pair_id = random.sample(self.class_idxs[neg_class], k=1)[0]

        return pair_id, is_pos

    def which_class(self):
        is_pos = random.uniform(0,1)
        if is_pos < self.pos_threshold:
            return 1
        return 0


class TripletSampler(MetricSampler):
    def __init__(self, train_data):
        super().__init__(train_data)
    
    @property
    def is_triplet(self):
        return True

    def sample_data(self, anchor_id, anchor_target):
        pos_id = random.sample(self.class_idxs[anchor_target], k=1)[0]
        neg_class = random.choice(
                [x for x in self.classes_list if x != anchor_target])
        neg_id = random.sample(self.class_idxs[neg_class], k=1)[0]

        return pos_id, neg_id

if __name__ == "__main__":
    data_path = make_directory("../datasets/")
    mnist_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = MNIST(root=data_path, train=True, transform=mnist_transforms)

    sampler = TripletSampler(train_data)
    print(sampler.is_triplet)
    print(sampler.sample_data(0, 5))

