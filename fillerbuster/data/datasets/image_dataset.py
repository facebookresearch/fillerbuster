"""
Image dataset.
Returns a batch of images, but not multi-view. This should be indicated in the returned data.
Images returned are 3xHxW in the range [0, 1].
# TODO: implement this file
"""

from torch.utils.data import IterableDataset


class ImageDataset(IterableDataset):
    """Image Dataset
    """

    def __init__(
        self,
        local_rank: int = 0,
        global_rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        shuffle: bool = True,
        size: int = 512,
        use_gpu: bool = True,
    ):
        super().__init__()

    def __iter__(self):
        raise NotImplementedError("Use the collate function instead.")
