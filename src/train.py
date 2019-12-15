
import torch
from tqdm import tqdm


def train(model, criterion, train_dataloader, optimizer, scheduler, n_epochs=2):
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, criterion, optimizer, train_dataloader)
        print("Epoch{} loss{}".format(epoch, train_loss))
        scheduler.step()


def train_epoch(model, criterion, optimizer, dataloader):
    total_loss = 0
    for idx, (anchor, duplet, is_pos) in enumerate(tqdm(dataloader)):
        output1, output2 = model(anchor, duplet)
        loss = criterion(output1, output2, is_pos)
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

    total_loss /= len(dataloader)
    return total_loss

if __name__ == "__main__":
    train(1)
        









