import torch
from tqdm import tqdm

from config import device
from utils import save_model, send_to_device
from metrics import mean_average_precision

from torch import autograd

def train(model, criterion, train_loader, query_loader, gallery_loader, optimizer, scheduler, experiment_name, sampling_method, n_epochs=2):
    for epoch in range(n_epochs):
        train_loss, active_samples = train_epoch(model, criterion, optimizer, train_loader, sampling_method)
        print("Epoch: {} loss: {} Active samples: {}".format(epoch, train_loss, active_samples))
        # scheduler.step()
        if epoch % 10 == 0:
            mean_avg_precision = mean_average_precision(model, query_loader, gallery_loader, k=100)
            print(mean_avg_precision)

    save_model(model, experiment_name)
    mean_avg_precision = mean_average_precision(model, query_loader, gallery_loader, k=100)
    print(mean_avg_precision)


def train_epoch(model, criterion, optimizer, dataloader, sampling_method):
    model.train()
    total_loss = 0
    active_samples = 0
    for idx, data_items in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        data_items = send_to_device(data_items, device)

        if sampling_method == "contrastive":
            output1, output2 = model(data_items["anchor"], data_items["duplet"])
            loss, active_sample = criterion(output1, output2, data_items["is_pos"])
        elif sampling_method == "triplet":
            anchor, pos, neg = model(data_items["anchor"], data_items["pos"], data_items["neg"])
            loss, active_sample = criterion(anchor, pos, neg, data_items["anchor_target"])
        elif sampling_method == "hardtriplet":
            anchor = model(data_items["anchor"])
            loss, active_sample = criterion(anchor, data_items["anchor_target"])

        total_loss += loss.item()
        active_samples += active_sample
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        
    total_loss /= len(dataloader)
    active_samples /= len(dataloader)
    return total_loss, active_samples


if __name__ == "__main__":
    train(1)
