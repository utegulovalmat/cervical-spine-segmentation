import os

import numpy as np
from torch.utils.data import Dataset as BaseDataset

from segmentation_models_pytorch.utils.custom_functions import read_pil_image


class MriDataset(BaseDataset):
    """
    # Usage example:

    %time
    get_training_augmentation = smp.utils.functions.get_training_augmentation
    get_test_augmentation = smp.utils.functions.get_test_augmentation

    train_dataset = MriDataset(path='...', augmentation=get_training_augmentation(), preprocessing=None)
    print(len(train_dataset))

    valid_dataset = MriDataset(path='...', augmentation=get_training_augmentation(), preprocessing=None)
    print(len(valid_dataset))

    test_dataset = MriDataset(path='...', augmentation=get_test_augmentation(), preprocessing=None)
    print(len(test_dataset))
    """

    CLASSES = ["1"]

    def __init__(
        self, path, augmentation=None, preprocessing=None,
    ):
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.exported_slices_dir = path

        all_fns = sorted(os.listdir(self.exported_slices_dir))
        self.image_fns = [fn for fn in all_fns if "seg" not in fn]
        self.mask_fns = [fn for fn in all_fns if "seg" in fn]
        assert len(self.mask_fns) == len(self.image_fns)

        self.slices_cnt = len(self.mask_fns)

    def __getitem__(self, image_id):
        image = self.load_image(image_id)
        image = np.expand_dims(image, axis=-1)  # add unit dimension
        mask = self.load_mask(image_id)
        mask = np.expand_dims(mask, axis=-1)  # add unit dimension

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def load_mask(self, image_id):
        mask_fn = self.exported_slices_dir + self.mask_fns[image_id]
        mask = read_pil_image(mask_fn)
        mask[mask > 0] = 1
        return mask.astype("uint8").T

    def load_image(self, image_id):
        image_fn = self.exported_slices_dir + self.image_fns[image_id]
        image = read_pil_image(image_fn)
        return image.T

    def __len__(self):
        return self.slices_cnt
