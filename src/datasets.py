import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

from utils import make_directory


class Cars3D:
    def __init__(self, root, mode, transform=None, train_size=100, image_size=32, query_split=10):
        self.data_path = os.path.join(root, "Cars3D", "images")
        self.train = True if mode == "train" else False
        self.mode = mode
        self.transform = transform
        self.train_size = train_size
        self.query_split = query_split

        self.tensor_transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()])

        self.data_files = self.read_data()
        self.data, self.targets = self.load_data()

    def read_data(self):
        cars = [f for f in os.listdir(self.data_path) if f.startswith("car")]
        cars.sort()  # sort to get always the same train/test set
        self.map_car2idx(cars)
        if self.train:
            cars = cars[:self.train_size]
        else:
            cars = cars[self.train_size:]
        return self.read_all_files(cars)

    def read_all_files(self, paths):
        all_cars = []
        for path in paths:
            temp_path = os.path.join(self.data_path, path)
            cars = [os.path.join(temp_path, f) for f in os.listdir(
                os.path.join(temp_path)) if f.startswith("car")]
            if self.mode=="train":
                all_cars += cars
            elif self.mode=="query":
                all_cars += cars[:self.query_split]
            elif self.mode=="gallery":
                all_cars += cars[self.query_split:]
        return all_cars

    def map_car2idx(self, cars):
        self.car2idx = {}
        self.idx2car = {}
        for idx, car in enumerate(cars):
            self.car2idx[car] = idx
            self.idx2car[idx] = car

    def load_data(self):
        data = []
        targets = []
        for data_point in self.data_files:
            target = self.car2idx[data_point.split("/")[4]]  # map car_id_mesh to idx
            image = Image.open(data_point)
            image = self.process_data(image, target)

            data.append(image)
            targets.append(target)

        return torch.cat(data), targets

    def process_data(self, image, target):
        image = self.tensor_transform(image)
        return image.unsqueeze(dim=0)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __len__(self):
        return len(self.data_files)


if __name__ == "__main__":
    data_path = make_directory("../datasets/")
    train_data = Cars3D(data_path, mode="query")
    print(len(train_data))
    print(train_data.data_files)
    # test_data = Cars3D(data_path, train=False, transform=None)
