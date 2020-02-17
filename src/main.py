import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch
from collections import OrderedDict
import uuid

from BaseData import BaseData
from OnlineSampler import OnlineSampler
from datasets import *
from losses import *
from networks import *

from train import train
from metrics import evaluation, update_metrics, write_results
import utils
from utils import device

args = utils.config()
print(args)

data_path = utils.make_directory(args.data_path)

experiment_id = str(uuid.uuid4().fields[-1])[:5]
experiment_name = args.dataset + "_" + \
    str(args.embedding_dim) + "_" + args.sampling_method + "_" + experiment_id
print(experiment_name)

if args.dataset == "Cars3D":
    train_data = Cars3D(root=args.data_path, mode="train")
    query_data = Cars3D(root=args.data_path, mode="query")
    gallery_data = Cars3D(root=args.data_path, mode="gallery")
elif args.dataset == "CarsEPFL":
    train_data = CarsEPFL(root=args.data_path, mode="train")
    query_data = CarsEPFL(root=args.data_path, mode="query")
    gallery_data = CarsEPFL(root=args.data_path, mode="gallery")
elif args.dataset == "CarsVeri":
    train_data = CarsVeRi(root=args.data_path, mode="train")
    query_data = CarsVeRi(root=args.data_path, mode="query")
    gallery_data = CarsVeRi(root=args.data_path, mode="gallery")
elif args.dataset == "CarsStanford":
    train_data = CarsStanford(root=args.data_path, mode="train")
    query_data = CarsStanford(root=args.data_path, mode="query")
    gallery_data = CarsStanford(root=args.data_path, mode="gallery")
else:
    raise ValueError("Provided dataset does not exist")

embedding_net = CIFAREmbeddingNet(args.embedding_dim)

if args.sampling_method == "contrastive":
    criterion = ContrastiveLoss(margin=args.margin)
    model = SiameseNet(embedding_net).to(device)
elif args.sampling_method == "triplet":
    criterion = TripletLoss(margin=args.margin)
    model = TripletNet(embedding_net).to(device)
elif args.sampling_method == "batch_soft":
    # very small temperature TODO:check performance
    criterion = BatchSoft(margin=args.margin,T=0.01)
    model = embedding_net.to(device)
elif args.sampling_method == "batch_hard":
    criterion = BatchHard(margin=args.margin)
    model = embedding_net.to(device)
else:
    raise NotImplementedError


train_dataset = BaseData(train_data, args.sampling_method)
query_dataset = BaseData(query_data, args.sampling_method)
gallery_dataset = BaseData(gallery_data, args.sampling_method)

if args.sampling_method == "batch_hard" or "batch_soft":
    balanced_sampler = OnlineSampler(train_dataset.targets, n_classes=args.n_classes, n_samples=args.n_samples)
    train_loader = DataLoader(train_dataset, batch_sampler=balanced_sampler)
else:
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

query_loader = DataLoader(
    query_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
gallery_loader = DataLoader(
    gallery_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size)

train(model, criterion, train_loader, query_loader, gallery_loader, optimizer, scheduler, experiment_name, args.sampling_method, n_epochs=args.n_epochs)

maps = OrderedDict(
    {"map_1": [], "map_5": [], "map_50": [], "map_100": [], "map_gallery": []})
hits = OrderedDict(
    {"hit_1": [], "hit_5": [], "hit_50": [], "hit_100": [], "hit_gallery": []})
recalls = OrderedDict(
    {"recall_1": [], "recall_5": [], "recall_50": [], "recall_100": [], "recall_gallery": []})

ks = evaluation(model, query_loader, gallery_loader)
maps, hits, recalls = update_metrics(ks, maps, hits, recalls)
write_results({"map": maps, "hits": hits, "recalls": recalls}, experiment_name)
