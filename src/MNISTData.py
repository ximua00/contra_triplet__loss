from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from ContrastiveSampler import ContrastiveSampler
from utils import make_directory


data_path = make_directory("../datasets/")


class MNISTData(Dataset):
    def __init__(self, data_path, sampling_method="contrastive"):
        self.data_path = data_path
        self.sampling_method = sampling_method
        self.load_data()

        if self.sampling_method == "contrastive":
            sampler = ContrastiveSampler(self.train_data)
        self.sampled_train_idxs = sampler.sample_data()

    def load_data(self):
        self.train_data = MNIST(root=data_path, train=True)
        self.test_data = MNIST(root=data_path, train=False)

    def __getitem__(self, idx):
        anchor_id, duplet_id, is_pos = self.sampled_train_idxs[idx]
        anchor_img = self.train_data.data[anchor_id]/255.0
        duplet_img = self.train_data.data[duplet_id]/255.0

        return anchor_img, duplet_img, is_pos

    def __len__(self):
        return len(self.sampled_train_idxs)

    def show_image(self, idx):
        im = self.train_data.data[idx]
        trans = transforms.ToPILImage()
        im = trans(im)
        im.show()



if __name__ == "__main__":
    data = MNISTData(data_path)
    dataloader = DataLoader(data, batch_size=4)
    for anchor, duplet, is_pos in dataloader:
        print(anchor.size())
        break
