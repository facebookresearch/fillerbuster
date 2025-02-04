"""
We use this class to load all our datasets, both single-view and multi-view.
"""

import random
from typing import Literal, Optional, Tuple

import torch

from fillerbuster.data.datasets.multi_dataset import MultiDataset
from fillerbuster.data.datasets.scannetpp_dataset import ScannetppDataset
from fillerbuster.data.datasets.shutterstock3d_dataset import Shutterstock3DDataset


class FillerbusterData:
    """
    Our dataloader supports training on both images and multi-view images.
    Args:
        ms_dataset_ratio - A tuple of size two, where the first element is for sampling a multi-view batch.
            The second is for sampling a single-view batch.
    """

    def __init__(
        self,
        local_rank: int = 0,
        global_rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        shuffle: bool = True,
        patch_size: int = 512,
        num_patches: int = 4,
        percent_force_one_patch_per_image: float = 0.9,
        percent_force_fixed_location: float = 0.1,
        percent_force_center_crop: float = 0.1,
        use_gpu: bool = True,
        mview_batch_size: int = 4,
        sview_batch_size: int = 1,
        num_workers: int = 1,
        ms_dataset_ratio: Optional[Tuple[int]] = None,
        mprobs: Optional[Tuple[float]] = (0.2, 0.1, 0.4, 0.3),
        shutterstock3d_folder: str = "/gen_ca/data/shutterstock_3d/",
        scannetpp_folder: str = "/gen_ca/data/scannetpp_v1/data/",
        dl3dv_folder: str = "/gen_ca/data/DL3DV-10K/DL3DV-ALL-960P/",
        mvimgnet_folder: str = "/gen_ca/data/mvimgnet/",
        use_ray_augmentation: bool = False,
        ray_augmentation_center_mode: Literal["random", "camera"] = "camera",
        ray_augmentation_rotate_mode: Literal["random", "camera"] = "camera",
        log_folder: Optional[str] = None,
    ):
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.percent_force_one_patch_per_image = percent_force_one_patch_per_image
        self.percent_force_fixed_location = percent_force_fixed_location
        self.percent_force_center_crop = percent_force_center_crop
        self.use_gpu = use_gpu
        self.mview_batch_size = mview_batch_size
        self.sview_batch_size = sview_batch_size
        self.num_workers = num_workers
        self.ms_dataset_ratio = ms_dataset_ratio
        self.mprobs = mprobs
        self.shutterstock3d_folder = shutterstock3d_folder
        self.scannetpp_folder = scannetpp_folder
        self.dl3dv_folder = dl3dv_folder
        self.mvimgnet_folder = mvimgnet_folder
        self.use_ray_augmentation = use_ray_augmentation
        self.ray_augmentation_center_mode = ray_augmentation_center_mode
        self.ray_augmentation_rotate_mode = ray_augmentation_rotate_mode
        self.log_folder = log_folder

        # multi-view dataset and dataloader
        self.multi_view_iter = None
        if self.ms_dataset_ratio[0] > 0:
            self.multi_view_dataset = self.build_multi_view_dataset()
            self.multi_view_dataloader = torch.utils.data.DataLoader(
                self.multi_view_dataset,
                batch_size=self.mview_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
            )
            self.multi_view_iter = iter(self.multi_view_dataloader)

        # single-view dataset and dataloader
        self.single_view_iter = None
        if self.ms_dataset_ratio[1] > 0:
            self.single_view_dataset = self.build_single_view_dataset()
            self.single_view_dataloader = torch.utils.data.DataLoader(
                self.single_view_dataset,
                batch_size=self.sview_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
            )
            self.single_view_iter = iter(self.single_view_dataloader)

    def build_multi_view_dataset(self):
        """
        Build a dataset for multi-view images.
        """
        dataset_s3 = Shutterstock3DDataset(
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            seed=self.seed,
            folder=self.shutterstock3d_folder,
            shuffle=self.shuffle,
            num_patches=self.num_patches,
            percent_force_one_patch_per_image=self.percent_force_one_patch_per_image,
            percent_force_fixed_location=self.percent_force_fixed_location,
            percent_force_center_crop=self.percent_force_center_crop,
            patch_size=self.patch_size,
            camera_res_scale_factor=1.0 * self.patch_size / 512,
            use_gpu=self.use_gpu,
            use_ray_augmentation=self.use_ray_augmentation,
            ray_augmentation_center_mode=self.ray_augmentation_center_mode,
            ray_augmentation_rotate_mode=self.ray_augmentation_rotate_mode,
            log_folder=self.log_folder,
        )
        dataset_sc = ScannetppDataset(
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            seed=self.seed,
            folder=self.scannetpp_folder,
            shuffle=self.shuffle,
            num_patches=self.num_patches,
            percent_force_one_patch_per_image=self.percent_force_one_patch_per_image,
            percent_force_fixed_location=self.percent_force_fixed_location,
            percent_force_center_crop=self.percent_force_center_crop,
            patch_size=self.patch_size,
            camera_res_scale_factor=0.5 * self.patch_size / 512,
            use_gpu=self.use_gpu,
            use_ray_augmentation=self.use_ray_augmentation,
            ray_augmentation_center_mode=self.ray_augmentation_center_mode,
            ray_augmentation_rotate_mode=self.ray_augmentation_rotate_mode,
            log_folder=self.log_folder,
        )
        dataset = MultiDataset(datasets=[dataset_s3, dataset_sc], probs=self.mprobs)
        return dataset

    def build_single_view_dataset(self):
        """
        Build a dataset for single-view images.
        """
        from fillerbuster.data.datasets.image_dataset import ImageDataset

        dataset_s2 = ImageDataset(
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            seed=self.seed,
            shuffle=self.shuffle,
            size=self.patch_size,
            use_gpu=False,  # We don't use GPU for single-view images.
        )
        return dataset_s2

    def __iter__(self):
        return self

    def __next__(self):
        """
        This next function is not synchronzed for a given a step. When using multiple GPUs,
        a given GPU will get either a single-view or multi-view batch. Use synchronized_next()
        to make all the GPUs on a given step get either a single-view or multi-view batch.
        """
        total = sum(self.ms_dataset_ratio)
        ms_dataset_probs = (self.ms_dataset_ratio[0] / total, self.ms_dataset_ratio[1] / total)
        data_iter = random.choices(
            population=[self.multi_view_iter, self.single_view_iter], weights=ms_dataset_probs, k=1
        )[0]
        return next(data_iter)

    def synchronized_next(self, step: int):
        """
        Synchronized next function will return all single-view or all multi-view images for a given step.
        This will not mix the two types in order to have maximum speed during training.
        This is because single-view batches will be processed faster than multi-view batches.
        """
        total = sum(self.ms_dataset_ratio)
        if step % total < self.ms_dataset_ratio[0]:
            return next(self.multi_view_iter)
        else:
            return next(self.single_view_iter)

    def get_total_samples(self):
        num_total_samples = 0
        if self.ms_dataset_ratio[0] > 0:
            num_total_samples += self.multi_view_dataset.get_total_samples()
        if self.ms_dataset_ratio[1] > 0:
            num_total_samples += self.single_view_dataset.get_total_samples()
        return num_total_samples
