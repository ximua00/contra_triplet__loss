import torch
from tqdm import tqdm

from utils import save_model
from metrics import mean_average_precision


def train(model, criterion, train_dataloader, test_dataloader, optimizer, scheduler, experiment_name, n_epochs=2):
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, criterion, optimizer, train_dataloader)
        mean_avg_precision = mean_average_precision(model, test_dataloader, train_dataloader, k=100)
        print("Epoch {} loss {} mAP {}".format(epoch, train_loss, mean_avg_precision))
        scheduler.step()
    save_model(model, experiment_name)


def train_epoch(model, criterion, optimizer, dataloader):
    model.train()
    total_loss = 0
    for idx, data_items in enumerate(tqdm(dataloader)):
        output1, output2 = model(data_items["anchor"], data_items["duplet"])
        loss = criterion(output1, output2, data_items["is_pos"])
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

    total_loss /= len(dataloader)
    return total_loss


if __name__ == "__main__":
    train(1)
        








