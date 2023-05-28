import torch
from torchmetrics import PeakSignalNoiseRatio as _PSNR, StructuralSimilarityIndexMeasure as _SSIM
from torchmetrics.image.fid import FrechetInceptionDistance as _FID
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as _LPIPS
from .utils import from_uint8_tensor

class PSNR(_PSNR):
    def update(self, preds, target):
        super().update(preds.float(), target.float())


class SSIM(_SSIM):
    def update(self, preds, target):
        super().update(preds.float(), target.float())


class FID(_FID):
    def update(self, preds, target):
        if self.reset_real_features:
            super().update(target.expand(-1, 3, -1, -1), real=True) 
        super().update(preds.expand(-1, 3, -1, -1), real=False)


class LPIPS(_LPIPS):
    def update(self, preds, target):
        preds = from_uint8_tensor(preds)
        target = from_uint8_tensor(target)
        super().update(target.expand(-1, 3, -1, -1), preds.expand(-1, 3, -1, -1))


if __name__ == "__main__":
    PSNR()
    SSIM()
    FID()