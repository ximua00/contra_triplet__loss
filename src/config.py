import torch

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
