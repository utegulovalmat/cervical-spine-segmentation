import os
import cv2
import random
import shutil
import albumentations as A
import matplotlib.pyplot as plt
import nrrd
import numpy as np
from PIL import Image
import SimpleITK as sitk
from skimage import morphology

ORIENTATION = {"coronal": "COR", "axial": "AXI", "sagital": "SAG"}


def combine_masks(image, mask):
    """
    Overlay mask layers into single mask
    Add mask to the image into single image
    """
    masksum = np.zeros(image.shape[:2])
    masked = np.zeros(image.shape[:2])
    for idx in range(mask.shape[2]):
        masksum += mask[:, :, idx] * 2 ** (idx + 1)
        masked += image[:, :, 0] * mask[:, :, idx]
    return masksum, masked


def get_train_augmentation_hardcore(hw_len=512):
    transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.05, rotate_limit=15, p=0.5
        ),
        A.OneOf(
            [
                A.GaussNoise(var_limit=0.01, p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
                A.Blur(blur_limit=3, p=0.5),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.GridDistortion(
                    distort_limit=0.3, interpolation=cv2.INTER_NEAREST, p=0.5
                ),
                A.OpticalDistortion(p=0.5),
                A.ElasticTransform(p=0.5),
            ],
            p=0.5,
        ),
        A.PadIfNeeded(
            min_height=hw_len,
            min_width=hw_len,
            always_apply=True,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.CenterCrop(height=hw_len, width=hw_len, always_apply=True),
    ]
    return A.Compose(transform, p=1)


def get_train_augmentation_medium(hw_len=512):
    transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.GridDistortion(
            num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_NEAREST
        ),
        A.Blur(blur_limit=3),
        # A.ElasticTransform(),
        # A.CoarseDropout(),
        A.PadIfNeeded(
            min_height=hw_len,
            min_width=hw_len,
            always_apply=True,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.CenterCrop(height=hw_len, width=hw_len, always_apply=True),
    ]
    return A.Compose(transform, p=1)


def get_train_augmentation_low(hw_len=512):
    transform = [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomScale(p=0.5),
        A.PadIfNeeded(
            min_height=hw_len,
            min_width=hw_len,
            always_apply=True,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.CenterCrop(height=hw_len, width=hw_len, always_apply=True),
    ]
    return A.Compose(transform)


def get_test_augmentation(hw_len=512):
    transform = [
        A.PadIfNeeded(
            min_height=hw_len,
            min_width=hw_len,
            always_apply=True,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.CenterCrop(height=hw_len, width=hw_len, always_apply=True),
    ]
    return A.Compose(transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        # A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)


def visualize(output_path, **images):
    """Plot images in one row.
    Helper function for data visualization"""
    n = len(images)
    plt.figure(figsize=(10, 5))
    for idx, (name, image) in enumerate(images.items()):
        image = np.ma.masked_where(image == 0, image)
        plt.subplot(1, n, idx + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image, alpha=0.5)
    # plt.axis("off")
    # plt.show()
    plt.savefig(fname=output_path)
    plt.close()


def plot_masks_overlay(output_path, image, gt_mask, pr_mask):
    cmaps = ["gray", "Set2", "Dark2"]
    f, ax = plt.subplots(1, 1, figsize=(10, 10))

    gt_mask = np.ma.masked_where(gt_mask == 0, gt_mask)
    countour = np.logical_xor(gt_mask, morphology.binary_erosion(gt_mask))
    image[countour > 0] = 1

    orig_pr_mask = pr_mask.copy()
    orig_pr_mask = np.ma.masked_where(orig_pr_mask == 0, orig_pr_mask)

    pr_mask[pr_mask == 1] = 2
    pr_mask = np.ma.masked_where(pr_mask == 0, pr_mask)
    pr_countour = np.logical_xor(pr_mask, morphology.binary_erosion(pr_mask))
    pr_mask[pr_countour > 0] = 1
    pr_mask[pr_mask == 2] = np.nan

    plt.imshow(image, alpha=1, cmap=cmaps[0])
    # plt.imshow(gt_mask, alpha=0.5, cmap=cmaps[1])
    plt.imshow(orig_pr_mask, alpha=0.3, cmap=cmaps[1])
    plt.imshow(pr_mask, alpha=1, cmap=cmaps[1])

    plt.axis("off")
    f.tight_layout()
    # plt.show()
    plt.savefig(fname=output_path)
    plt.close(f)


def rotate_orientation(volume_data, volume_label, orientation=ORIENTATION["coronal"]):
    """Return rotated matrix to get differnt views ralative to submited 3D volumes"""
    if orientation == ORIENTATION["coronal"]:
        return volume_data.transpose((2, 0, 1)), volume_label.transpose((2, 0, 1))
    elif orientation == ORIENTATION["axial"]:
        return volume_data.transpose((1, 2, 0)), volume_label.transpose((1, 2, 0))
    elif orientation == ORIENTATION["sagital"]:
        return volume_data, volume_label
    else:
        raise ValueError("Invalid value for orientation. Pleas see help")


def remove_black(data, labels, only_with_target=False):
    clean_data, clean_labels = [], []
    for i, frame in enumerate(labels):
        unique, counts = np.unique(frame, return_counts=True)
        # if only_with_target and len(unique) == 1:
        #    continue
        if counts[0] / sum(counts) < 0.99:
            clean_labels.append(frame)
            clean_data.append(data[i])
    return np.array(clean_data), np.array(clean_labels)


def normilize_mean_std(volume):
    # now normalize each modality with its mean and standard deviation (computed within the brain mask)
    mean = volume.mean()
    std = volume.std()
    volume = (volume - mean) / (std + 1e-8)
    # imgs_npy[i][brain_mask == 0] = 0
    return volume


def remove_all_blacks(image, mask, only_with_target=False):
    image, mask = rotate_orientation(image, mask, orientation=ORIENTATION["coronal"])
    image, mask = remove_black(image, mask, only_with_target)
    image, mask = rotate_orientation(image, mask, orientation=ORIENTATION["axial"])
    image, mask = remove_black(image, mask, only_with_target)
    image, mask = rotate_orientation(image, mask, orientation=ORIENTATION["sagital"])
    image, mask = remove_black(image, mask, only_with_target)
    return image, mask


def round_clip_0_1(x):
    """Remove values gt 1 and lt 0"""
    return x.round().clip(0, 1)


def normalize_0_1(x):
    x_max = np.max(x)
    x_min = np.min(x)
    # x_max = np.percentile(x, 99)
    # x_min = np.percentile(x, 1)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def read_volume(filepath):
    img_data, header = nrrd.read(filepath)
    return img_data


def read_volume_nifty(filepath):
    """For nii.gz data type
    pip install nibabel
    import nibabel as nib
    """
    # img = nib.load(filepath)
    # img = nib.as_closest_canonical(img)
    # img_data = img.get_fdata()
    # return img_data
    return


def read_pil_image(filepath):
    new_img = Image.open(filepath)
    new_img = np.array(new_img.getdata()).reshape(new_img.size[1], new_img.size[0])
    return new_img


def read_slices(images, masks):
    _images, _masks = [], []
    for image_fn, mask_fn in zip(images, masks):
        # Get volume and mask files by filepath
        image = read_volume(image_fn)
        mask = np.uint8(read_volume(mask_fn))
        mask[mask > 0] = 1

        # Remove black slices from all sides
        # image, mask = remove_all_blacks(image, mask, only_with_target=True)

        # N4 bias field correction
        # TODO: corrected volumes are exported to dataset_n4 folder
        # image = n4correction(image, mask, image_fn, mask_fn)
        # 0 ... 7
        # images before n4correction 745.6845776952546 470.2227819774101 0 4981
        # images after               745.62714         470.40427         0.0 5041.2744
        # 0 1
        # images before n4correction 745.6845776952546 470.22278 0   4981
        # images after               718.2532          429.12723 0.0 4265.9404

        # Z-score normalization
        image = zscore_normalize(image, mask)
        # 0-1
        image = normalize_0_1(image)

        _images.append(image)
        _masks.append(mask)
    return _images, _masks


def load_image_and_mask(
    slice_index, single_dimension, use_dimension, image_volumes, mask_volumes
):
    """ Extracts volume slice with index `slice_index`

    Volumes must be in corresponding order with masks.

    image_volumes:      3D numpy array
    mask_volumes:       3D numpy array
    single_dimension:   Use all 3D volume or 1 view dimension
                        True / False
    use_dimension:      Use only one of dimensions of 3D volume
                        can be use_dimension_0/use_dimension_1/use_dimension_2

    """
    image, mask = None, None
    for _image, _mask in zip(image_volumes, mask_volumes):
        if single_dimension:  # <-------------------------------- single dimension
            if use_dimension == "use_dimension_0":
                img_shape = _image.shape[0]
                if slice_index >= img_shape:
                    slice_index -= img_shape
                    continue
                if slice_index >= img_shape:
                    slice_index -= img_shape
                else:
                    image = _image[slice_index, :, :]
                    mask = _mask[slice_index, :, :]
                    break
            elif use_dimension == "use_dimension_1":
                img_shape = _image.shape[1]
                if slice_index >= img_shape:
                    slice_index -= img_shape
                    continue
                if slice_index >= img_shape:
                    slice_index -= img_shape
                else:
                    image = _image[:, slice_index, :]
                    mask = _mask[:, slice_index, :]
                    break
            elif use_dimension == "use_dimension_2":
                img_shape = _image.shape[2]
                if slice_index >= img_shape:
                    slice_index -= img_shape
                    continue
                if slice_index >= img_shape:
                    slice_index -= img_shape
                else:
                    image = _image[:, :, slice_index]
                    mask = _mask[:, :, slice_index]
                    break
        else:  # <---------------------------------------------------- 3 dimensions
            img_shape = _image.shape
            # Get target volume
            if slice_index >= sum(img_shape):
                slice_index -= sum(img_shape)
                continue
            # Get target dimension and slice
            if slice_index >= img_shape[0]:
                slice_index -= img_shape[0]
            else:
                image = _image[slice_index, :, :]
                mask = _mask[slice_index, :, :]
                break
            if slice_index >= img_shape[1]:
                slice_index -= img_shape[1]
            else:
                image = _image[:, slice_index, :]
                mask = _mask[:, slice_index, :]
                break

            if slice_index > img_shape[2]:
                slice_index -= img_shape[2]
            else:
                image = _image[:, :, slice_index]
                mask = _mask[:, :, slice_index]
                break
    return image, mask


def save_slice_as_tiff_image(
    npy_orig, convert_format, output_dir: str, new_title: str, sanity_check=False
):
    """Export numpy array to TIFF image.

    npy_orig:       numpy array containing image
    convert_format: must be 'F' for grayscale, 'L' for int values
    """
    assert new_title.endswith(".tiff")

    out_fname = output_dir + new_title
    new_img = Image.fromarray(npy_orig)
    new_img = new_img.convert(convert_format)
    new_img.save(out_fname)
    new_img = read_pil_image(out_fname)
    assert new_img.shape == npy_orig.shape

    if sanity_check:
        new_img = read_pil_image(out_fname)
        print(out_fname)
        print(np.sum(npy_orig), "<sum>", np.sum(new_img))
        print(np.unique(npy_orig), "<np.unique>", np.unique(new_img))
        print(np.max(npy_orig), "<max>", np.max(npy_orig))
        print(np.min(new_img), "<min>", np.min(new_img))
        print(type(npy_orig), "<type>", type(new_img))
        print(npy_orig.dtype, "<dtype>", new_img.dtype)
        print(npy_orig.shape, "<shape>", new_img.shape)
        print(npy_orig)
        print(new_img)
        print("\n")

    return npy_orig, new_img, out_fname


def extract_slices_from_volumes(
    images, masks, output_dir, skip_empty_mask=True, use_dimensions="012",
):
    """ Export volumes slices as separate TIFF images

    :param images: list of volume paths
    :param masks: list of volume paths
    :param output_dir: target folder path suffix
    :param skip_empty_mask: default True
    :param use_dimensions: which views to extract "012"

    # Usage example

    EXPORTED_SLICES_DIR = '/content/export_slices/'
    if os.path.isdir(EXPORTED_SLICES_DIR):
        print('rmtree folder', EXPORTED_SLICES_DIR)
        shutil.rmtree(EXPORTED_SLICES_DIR)
    os.mkdir(EXPORTED_SLICES_DIR)
    extract_slices_from_volumes(TRAIN, TRAIN_MASKS, output_dir=EXPORTED_SLICES_DIR)
    """
    if os.path.isdir(output_dir):
        print("rmtree before extracting slices:", output_dir)
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    image_volumes, mask_volumes = read_slices(images, masks)
    volume_shapes = [i.shape for i in image_volumes]
    print("volumes shapes", volume_shapes)
    print("skip empty masks", skip_empty_mask)

    # Export slices from dimension 0
    slices_cnt_dim_0 = sum([x for x, y, z in volume_shapes])
    with_masks_dim_0 = slices_cnt_dim_0
    if "0" in use_dimensions:
        for idx in range(0, slices_cnt_dim_0):
            npy_image, npy_mask = load_image_and_mask(
                idx,
                single_dimension=True,
                use_dimension="use_dimension_0",
                image_volumes=image_volumes,
                mask_volumes=mask_volumes,
            )
            if skip_empty_mask and len(np.unique(npy_mask)) == 1:
                with_masks_dim_0 -= 1
                continue
            base_fn = str(idx).zfill(7)
            image_fn = base_fn + ".tiff"
            mask_fn = base_fn + "_seg.tiff"
            save_slice_as_tiff_image(
                npy_image, convert_format="F", output_dir=output_dir, new_title=image_fn
            )
            save_slice_as_tiff_image(
                npy_mask, convert_format="L", output_dir=output_dir, new_title=mask_fn
            )
        print("exported slices dim 0:", slices_cnt_dim_0)
        print("      with mask dim 0:", with_masks_dim_0)

    # Export slices from dimension 1
    start_idx = slices_cnt_dim_0
    slices_cnt_dim_1 = sum([y for x, y, z in volume_shapes])
    with_masks_dim_1 = slices_cnt_dim_1
    if "1" in use_dimensions:
        for idx in range(0, slices_cnt_dim_1):
            npy_image, npy_mask = load_image_and_mask(
                idx,
                single_dimension=True,
                use_dimension="use_dimension_1",
                image_volumes=image_volumes,
                mask_volumes=mask_volumes,
            )
            if skip_empty_mask and len(np.unique(npy_mask)) == 1:
                with_masks_dim_1 -= 1
                continue
            base_fn = str(idx + start_idx).zfill(7)
            image_fn = base_fn + ".tiff"
            mask_fn = base_fn + "_seg.tiff"
            save_slice_as_tiff_image(
                npy_image, convert_format="F", output_dir=output_dir, new_title=image_fn
            )
            save_slice_as_tiff_image(
                npy_mask, convert_format="L", output_dir=output_dir, new_title=mask_fn
            )
        print("exported slices dim 1:", slices_cnt_dim_1)
        print("      with mask dim 1:", with_masks_dim_1)

    # Export slices from dimension 2
    start_idx = slices_cnt_dim_0 + slices_cnt_dim_1
    slices_cnt_dim_2 = sum([z for x, y, z in volume_shapes])
    with_masks_dim_2 = slices_cnt_dim_2
    if "2" in use_dimensions:
        for idx in range(0, slices_cnt_dim_2):
            npy_image, npy_mask = load_image_and_mask(
                idx,
                single_dimension=True,
                use_dimension="use_dimension_2",
                image_volumes=image_volumes,
                mask_volumes=mask_volumes,
            )
            if skip_empty_mask and len(np.unique(npy_mask)) == 1:
                with_masks_dim_2 -= 1
                continue
            base_fn = str(idx + start_idx).zfill(7)
            image_fn = base_fn + ".tiff"
            mask_fn = base_fn + "_seg.tiff"
            save_slice_as_tiff_image(
                npy_image, convert_format="F", output_dir=output_dir, new_title=image_fn
            )
            save_slice_as_tiff_image(
                npy_mask, convert_format="L", output_dir=output_dir, new_title=mask_fn
            )
        print("exported slices dim 2:", slices_cnt_dim_2)
        print("      with mask dim 2:", with_masks_dim_2)
    return True


def set_global_seed(seed: int) -> None:
    """
    Sets random seed into PyTorch, TensorFlow, Numpy and Random.
    Args:
        seed: random seed
    """
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.set_printoptions(precision=10)
    random.seed(seed)
    np.random.seed(seed)
    # cudnn.benchmark = False
    # cudnn.deterministic = True


def export_slices_to_volume(fn, slices, shape):
    # volume = np.dstack(slices)
    volume = np.stack(slices)
    nrrd.write(fn + "-corr.nrrd", volume)


def n4correction(input_img, mask, image_fn, mask_fn):
    """
    :param input_img: numpy array format
    :param mask: numpy array format
    :param image_fn: path to image volume
    :param mask_fn: path to mask volume
    :return: n4 bias field corrected image with numpy array format

    """
    input_image = sitk.GetImageFromArray(input_img)
    mask_image = sitk.GetImageFromArray(mask)
    input_image = sitk.Cast(input_image, sitk.sitkFloat32)
    mask_image = sitk.Cast(mask_image, sitk.sitkUInt8)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    image_corrected = corrector.Execute(input_image, mask_image)
    image_corrected = sitk.GetArrayFromImage(image_corrected)
    image_fn = image_fn.split(".")[0]
    mask_fn = mask_fn.split(".")[0]
    nrrd.write(image_fn + "-corr.nrrd", image_corrected)
    nrrd.write(mask_fn + "-seg-corr.nrrd", mask)
    return image_corrected


def zscore_normalize(image, mask):
    """
    https://github.com/jcreinhold/intensity-normalization/blob/master/intensity_normalization/normalize/zscore.py

    Normalize a target image by subtracting the mean of the vertebra
    and dividing by the standard deviation

    Args:
        image: target volume
        mask: mask for image
    Returns:
        normalized: image with mean at norm_value
    """
    logical_mask = mask == 1  # force the mask to be logical type
    mean = image[logical_mask].mean()
    std = image[logical_mask].std()
    normalized = (image - mean) / std
    return normalized


def count_hausdorff_distance():
    # Hausdorff distance calculation
    import itk

    tumor = itk.imread("tumor.nrrd")
    ablation = itk.imread("ablation.nrrd")

    a2t = itk.DirectedHausdorffDistanceImageFilter.New(ablation, tumor)
    t2a = itk.DirectedHausdorffDistanceImageFilter.New(tumor, ablation)

    a2t.Update()
    t2a.Update()

    print("Ablation to tumor: %f" % (a2t.GetDirectedHausdorffDistance()))
    print("Tumor to ablation: %f" % (t2a.GetDirectedHausdorffDistance()))
