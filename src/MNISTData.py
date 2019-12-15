from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from ContrastiveSampler import ContrastiveSampler
from utils import make_directory


data_path = make_directory("../datasets/")


class MNISTData(Dataset):
    def __init__(self, data, sampler):
        self.is_train = data.train
        self.sampler = sampler
        self.which_data(data)
        self.data_length = len(data)

    def which_data(self, data):
        if self.is_train:
            self.train_data = data
        else:
            self.test_data = data

    def __getitem__(self, idx):
        if self.is_train:
            anchor, anchor_target = self.train_data[idx]
            duplet_id, is_pos = self.sampler.sample_data(idx, anchor_target)
            duplet, _ = self.train_data[duplet_id]
            return anchor, duplet, is_pos
        else:
            raise NotImplementedError
        
    def __len__(self):
        return self.data_length

    def show_image(self, idx):
        im = self.train_data.data[idx]
        trans = transforms.ToPILImage()
        im = trans(im)
        im.show()



if __name__ == "__main__":
    mnist_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = MNIST(root=data_path, train=True, transform=mnist_transforms)
    train_dataset = MNISTData(train_data)
    
    dataloader = DataLoader(train_dataset, batch_size=4)
    for id, (anchor, duplet, is_pos) in enumerate(dataloader):
        # print(anchor.size())
        # print(duplet.size())
        # print(is_pos.size())
        # print(id)
        break
        
    