import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict

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
            if self.mode == "train":
                all_cars += cars
            elif self.mode == "query":
                all_cars += cars[:self.query_split]
            elif self.mode == "gallery":
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
            target = self.car2idx[data_point.split(
                "/")[5]]  # map car_id_mesh to idx
            image = Image.open(data_point)
            image = self.process_data(image)

            data.append(image)
            targets.append(target)

        return torch.cat(data), targets

    def process_data(self, image):
        image = self.tensor_transform(image)
        return image.unsqueeze(dim=0)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __len__(self):
        return len(self.data_files)


class CarsEPFL:
    def __init__(self, root, mode, transform=None, train_size=15, image_size=32, query_split=10):
        self.data_path = os.path.join(
            root, "epfl-multi-view-car", "tripod-seq")
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
        cars = [f for f in os.listdir(self.data_path) if f.endswith(".jpg")]
        cars.sort()

        car_paths = defaultdict(list)
        self.idx2car = dict()
        self.car2idx = dict()
        for idx, path in enumerate(cars):
            car = path.split("_")[2]
            car_paths[int(car)].append(path)
            self.idx2car[int(car)] = car
            self.car2idx[car] = int(car)

        return self.split_data(car_paths)

    def split_data(self, car_paths):
        data_files = []
        if self.mode == "train":
            for car_id in car_paths.keys():
                if car_id <= self.train_size:
                    data_files += car_paths[car_id]
        elif self.mode == "query":
            for car_id in car_paths.keys():
                if car_id > self.train_size:
                    data_files += car_paths[car_id][:self.query_split]
        elif self.mode == "gallery":
            for car_id in car_paths.keys():
                if car_id > self.train_size:
                    data_files += car_paths[car_id][self.query_split:]

        return data_files

    def load_data(self):
        data = []
        targets = []
        for data_point in self.data_files:
            target = self.car2idx[data_point.split(
                "_")[2]]  # map car_id_mesh to idx
            image = Image.open(os.path.join(self.data_path, data_point))
            image = self.process_data(image)
            data.append(image)
            targets.append(target)
        return torch.cat(data), targets

    def process_data(self, image):
        image = self.tensor_transform(image)
        return image.unsqueeze(dim=0)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __len__(self):
        return len(self.data_files)



class CarsShapeNet:
    def __init__(self, root, mode, transform=None, train_size=120, image_size=32, query_split=30):
        self.data_path = os.path.join(root, "shapenet_rendered")
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
        cars = [f for f in os.listdir(self.data_path) if f.startswith("models")]
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
                os.path.join(temp_path)) if f.startswith("models")]
            if self.mode == "train":
                all_cars += cars
            elif self.mode == "query":
                all_cars += cars[:self.query_split]
            elif self.mode == "gallery":
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
            target = self.car2idx[data_point.split(
                "/")[4]]  # map car_id_mesh to idx
            image = Image.open(data_point)
            image = image.convert("RGB")
            image = self.process_data(image)
            data.append(image)
            targets.append(target)

        return torch.cat(data), targets

    def process_data(self, image):
        image = self.tensor_transform(image)
        return image.unsqueeze(dim=0)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __len__(self):
        return len(self.data_files)


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from BaseData import BaseData

    sampling_method = "triplet"
    data_path = make_directory("../datasets/")
    train_data = CarsShapeNet(data_path, mode="train")
    train_dataset = BaseData(train_data, sampling_method=sampling_method)


    dataloader = DataLoader(train_dataset,batch_size=4, shuffle=True)
    for idx ,data_items in enumerate(dataloader):
        print(data_items["anchor_target"].size())
        print(data_items["anchor"].size())

        break

    

