import os
import numpy as np
import torch
import h5py


class DownscalerDataset(torch.utils.data.Dataset):
    def __init__(self, root, resolution=512, wavenumber=0, split="train", transform=None, target_transform=None):
        assert resolution in [64, 512]
        assert split in ["train", "test"]
        self.root = root
        # self.data = h5py.File(os.path.join(self.root, f"x{split}_{resolution}.jld2"), 'r')['single_stored_object']
        self.data = np.load(os.path.join(self.root, f"x{split}_{resolution}.npy"), mmap_mode='r')
        if resolution == 64:
            self.indices = np.arange(len(self.data))
        elif resolution == 512:
            if wavenumber == 0:
                self.indices = np.arange(len(self.data))
            else:
                assert wavenumber in [1, 2, 4, 8, 16]
                seg = int(np.log2(wavenumber))
                ndata_seg = 802 if split == 'train' else 200
                self.indices = np.arange(seg*ndata_seg, (seg+1)*ndata_seg)
        else:
            raise ValueError
        self.randperm = np.random.default_rng(42).permutation(len(self.indices))

        self.transform = transform
        self.target_transform = target_transform

        self.scaling = np.load(os.path.join(self.root, f"scaling_{resolution}.npz"))
        mintrain_mean, Delta_, mintrain_p, Delta_p = self.scaling['mintrain_mean'], self.scaling['Delta_'], self.scaling['mintrain_p'], self.scaling['Deltap']
        self.mintrain_mean, self.Delta_, self.mintrain_p, self.Delta_p = \
            mintrain_mean.transpose()[0], Delta_.transpose()[0], mintrain_p.transpose()[0], Delta_p.transpose()[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        rindex = self.indices[self.randperm[index]]
        data = self.data[rindex]
        img, targets = torch.from_numpy(data[..., :2, :, :]), torch.from_numpy(data[..., 2:, :, :])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, targets

    def invert_preprocessing(self, x_tilde, y_tilde=None):
        # x_tilde: (B,) C, H, W
        if y_tilde is None:
            y_tilde = torch.zeros_like(x_tilde)[..., 0:1, :, :]
        x_tilde = torch.cat([x_tilde, y_tilde], dim=-3)
        assert x_tilde.shape[-3] == 3
        tmp = (x_tilde + 2) / 2 * self.Delta_p + self.mintrain_p
        xp = tmp - tmp.mean((-2, -1), keepdims=True)
        x_bar = (tmp - xp) / self.Delta_p * self.Delta_ + self.mintrain_mean
        out = xp + x_bar
        return out[..., :2, :, :], out[..., 2:, :, :]

    def apply_preprocessing(self, x, y=None):
        # x: (B,) C, H, W
        if y is None:
            y = torch.zeros_like(x)[..., 0:1, :, :]
        x = torch.cat([x, y], dim=-3)
        assert x.shape[-3] == 3
        x_bar = x.mean((-2, -1), keepdims=True)
        xp = x - x_bar
        x_tilde = 2 * (x_bar - self.mintrain_mean) / self.Delta_ - 1
        x_tilde_p = 2 * (xp - self.mintrain_p) / self.Delta_p - 1
        out = x_tilde + x_tilde_p
        return out[..., :2, :, :], out[..., 2:, :, :]

