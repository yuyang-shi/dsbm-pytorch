import torch
from glob import glob
from pathlib import Path
from PIL import Image
import numpy as np


class AFHQ(torch.utils.data.Dataset):
    """
        root_dir is to train set that has the cat, dog, wild folders in 
        animal_type is either cat, dog, or wild
    """
    def __init__(self, root_dir, animal_type):
        self.root_dir = root_dir
        self.animal_type = animal_type
        assert animal_type in ['cat', 'dog', 'wild']
        self.all_image_paths = list(sorted(Path(self.root_dir).joinpath(animal_type).glob('*.png')))

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, index):
        path = self.all_image_paths[index]

        pil_image = Image.open(path)
        np_image = np.array(pil_image) # 0 to 255 integer

        # scale floats between -1 and 1
        tensor_image = (torch.tensor(np_image, dtype=torch.float32) / 255.0) * 2 - 1

        # current shape is (H, W, C)
        # transpose to (C, H, W)
        tensor_image = tensor_image.permute(2, 0, 1)

        return tensor_image, torch.zeros((1,))





