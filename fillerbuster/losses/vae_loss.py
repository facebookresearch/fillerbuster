"""
Losses for a VAE.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class VAELoss(nn.Module):
    """
    VAE losses.
    """

    def __init__(
        self,
        mse_scale: float = 1.0,
        lpips_scale: float = 1e-1,
        kl_scale: float = 1e-6,
    ):
        super().__init__()

        self.mse_scale = mse_scale
        self.lpips_scale = lpips_scale
        self.kl_scale = kl_scale

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", reduction="mean")
        self.psnr = PeakSignalNoiseRatio()

    def get_posterior_metrics(self, posterior):
        dict_m = {}
        dict_m["kl"] = posterior.kl().mean()
        return dict_m

    def get_recon_metrics(self, pred, pixel_values):
        dict_m = {}
        dict_m["mse"] = F.mse_loss(pred, pixel_values, reduction="mean")
        pred_clamp = torch.clamp(pred, min=-1.0, max=1.0)
        pixel_values_clamp = torch.clamp(pixel_values, min=-1.0, max=1.0)
        dict_m["lpips"] = self.lpips(pred_clamp, pixel_values_clamp)
        dict_m["psnr"] = self.psnr(pred, pixel_values)
        return dict_m

    def get_all_metrics(self, posterior, pred, pixel_values):
        posterior_metrics_dict = self.get_posterior_metrics(posterior)
        recon_metrics_dict = self.get_recon_metrics(pred, pixel_values)
        metrics_dict = {}
        metrics_dict.update(posterior_metrics_dict)
        metrics_dict.update(recon_metrics_dict)
        return metrics_dict

    def get_all_metrics_and_losses(self, posterior, pred, pixel_values):
        metrics_dict = self.get_all_metrics(posterior, pred, pixel_values)
        loss_dict = {}
        loss_dict["mse"] = metrics_dict["mse"] * self.mse_scale
        loss_dict["lpips"] = metrics_dict["lpips"] * self.lpips_scale
        loss_dict["kl"] = metrics_dict["kl"] * self.kl_scale
        return loss_dict, metrics_dict
