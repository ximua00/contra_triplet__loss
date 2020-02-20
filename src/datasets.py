import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
import pickle

from utils import make_directory

class CarsStanford:
    def __init__(self, root, mode, transform=None, train_size=120, image_size=32, query_split=10):
        self.data_path = os.path.join(root, "StanfordCars")
        self.mode = mode
        self.train = True if mode == "train" else False
        self.train_size = train_size
        self.query_split = query_split

        self.transform = transform
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.tensor_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                                 ])

        self.data_files = self.read_data()
        self.data, self.targets = self.load_data()

    def read_data(self):
        annotations_path = os.path.join(self.data_path, "car_annotations.pkl") 
        with open(annotations_path, "rb") as f:
             self.annotations = pickle.load(f)
        self.map_car2idx(self.annotations)
        self.__get_test_names(self.annotations)
        
        return self.read_all_files(list(self.annotations.keys()))
    
    def read_all_files(self, paths):
        all_cars = []
        for car in paths:
            car_model = self.annotations[car]["class"]
            if self.mode == "train":
                if car_model < self.train_size:
                    all_cars.append(car)    
            if self.mode == "query":
                if car in self.test_images[car_model][:self.query_split]:
                    all_cars.append(car)
            if self.mode == "gallery":
                if car in self.test_images[car_model][self.query_split:]:
                    all_cars.append(car)
        return all_cars

    def map_to_target(self, data_point):
        target = self.car2idx[self.annotations[data_point]["class"]]
        return target

    def load_data(self):
        data = []
        targets = []
        for data_point in self.data_files:
            target = self.map_to_target(data_point)
            image = Image.open(os.path.join(self.data_path, data_point))
            image = self.process_data(image, data_point)
            data.append(image)
            targets.append(target)
        return torch.cat(data), targets

    def __get_test_names(self, annotations):
        self.test_images = defaultdict(list)
        for car_image in annotations:
            car_model = annotations[car_image]["class"]
            if car_model>self.train_size:
                self.test_images[car_model].append(car_image)

    def map_car2idx(self, cars):
        self.car2idx = {}
        self.idx2car = {}
        for car_image in self.annotations.keys():
            car_class = self.annotations[car_image]["class"]
            self.car2idx[car_class] = car_class
            self.idx2car[car_class] = car_class
    
    def process_data(self, image, data_point):
        bbox = self.__get_bbox(data_point)
        image = image.crop(box = bbox)
        if image.getbands()[0] == 'L':
            image = image.convert('RGB')
        image = self.tensor_transform(image)
        return image.unsqueeze(dim=0)

    def __get_bbox(self, data_point):
        x1 = self.annotations[data_point]["bbox_x1"]
        y1 = self.annotations[data_point]["bbox_y1"]
        x2 = self.annotations[data_point]["bbox_x2"]
        y2 = self.annotations[data_point]["bbox_y2"]
        return x1, y1, x2, y2

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __len__(self):
        return len(self.data_files)


class CarsVeRi:
    def __init__(self, root, mode, transform=None, image_size=32):
        self.data_path = os.path.join(root, "VeRi_with_plate")
        self.mode = mode
        self.train = True if mode == "train" else False
        self.transform = transform
        self.mode_folder = {"train":"image_train", "query": "image_query", "gallery": "image_test"}
        self.tensor_transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()])

        self.data_files = self.read_data()
        self.data, self.targets = self.load_data()
        
    def read_data(self):
        self.train_cars = [f for f in os.listdir(os.path.join(self.data_path, self.mode_folder["train"])) if f.endswith(".jpg")]
        self.train_cars.sort()
        self.query_cars = [f for f in os.listdir(os.path.join(self.data_path, self.mode_folder["query"])) if f.endswith(".jpg")]
        self.query_cars.sort()
        self.gallery_cars = [f for f in os.listdir(os.path.join(self.data_path, self.mode_folder["gallery"])) if f.endswith(".jpg")]
        self.gallery_cars.sort()
        self.map_car2idx(self.train_cars + self.query_cars + self.gallery_cars)

        if not self.train:
            self.remove_excess_from_gallery()

        if self.mode == "train":
            return self.train_cars
        elif self.mode == "query":
            return self.query_cars
        elif self.mode == "gallery":
            return self.gallery_cars

    def remove_excess_from_gallery(self):
        for query_car in self.query_cars:
            if query_car in self.gallery_cars:
                self.gallery_cars.remove(query_car)

    def map_car2idx(self, cars):
        self.car2idx = {}
        self.idx2car = {}
        for idx, car in enumerate(cars):
            car_model = car.split("_")[0]
            self.car2idx[car_model] = idx
            self.idx2car[idx] = car_model

    def map_to_target(self, data_point):
        target = self.car2idx[data_point.split("_")[0]]
        return target

    def load_data(self):
        data = []
        targets = []
        for data_point in  self.data_files:
            target = self.map_to_target(data_point)
            image = Image.open(os.path.join(self.data_path ,self.mode_folder[self.mode],data_point))
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


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from BaseData import BaseData

    sampling_method = "triplet"
    data_path = make_directory("../../datasets/")
    train_data = CarsStanford(data_path, mode="train")
    # train_dataset = BaseData(train_data, sampling_method=sampling_method)


    # dataloader = DataLoader(train_dataset,batch_size=4, shuffle=True)
    # for idx ,data_items in enumerate(dataloader):
    #     print(data_items["anchor_target"].size())
    #     print(data_items["anchor"].size())

    #     break

    

