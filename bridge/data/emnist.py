import os, shutil
import urllib
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.utils import save_image


class FiveClassEMNIST(torchvision.datasets.EMNIST):
    def __init__(self, root="./data/emnist", train=True, download=True, transform=None, target_transform=None): 
        super().__init__(root=root, split="letters", train=train, download=download, transform=transform, target_transform=target_transform)
        self.custom_indices = (self.targets<=5).nonzero(as_tuple=True)[0]
        self.data, self.targets = self.data[self.custom_indices].transpose(1, 2), self.targets[self.custom_indices]
