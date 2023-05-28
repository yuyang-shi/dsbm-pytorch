import os
import torch
import torchvision.utils as vutils


def save_image(tensor, fp, format=None, **kwargs):
    normalized = normalize_tensor(tensor)
    vutils.save_image(normalized, fp, format=format, **kwargs)


def to_uint8_tensor(tensor):
    normalized = normalize_tensor(tensor)
    return normalized.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)


def from_uint8_tensor(tensor):
    normalized = tensor.float() / 255
    return unnormalize_tensor(normalized)


def normalize_tensor(tensor):
    normalized = tensor / 2 + 0.5
    return normalized.clamp_(0, 1)


def unnormalize_tensor(tensor):
    unnormalized = (tensor - 0.5) * 2
    return unnormalized.clamp_(-1, 1)


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results