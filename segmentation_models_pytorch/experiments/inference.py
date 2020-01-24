"""
    python -m segmentation_models_pytorch.experiments.inference
"""
import argparse
import os
import shutil

import matplotlib.pyplot as plt
import torch

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.experiments.helpers import plot_metrics_per_slice
from segmentation_models_pytorch.experiments.helpers import get_datetime_str
from segmentation_models_pytorch.utils.data import MriDataset
from segmentation_models_pytorch.utils.custom_functions import get_preprocessing
from segmentation_models_pytorch.utils.custom_functions import get_test_augmentation
from segmentation_models_pytorch.utils.custom_functions import read_pil_image


def export_volume_to_dir(
    patient: str, volume: str, mask: str, use_axis: str, skip_empty_mask: bool,
):
    print(patient)
    smp.utils.custom_functions.extract_slices_from_volumes(
        images=[volume],
        masks=[mask],
        output_dir=patient,
        skip_empty_mask=skip_empty_mask,
        use_dimensions=use_axis,
    )


def run_inference():
    input_dir = "/home/a/Thesis/datasets/mri/"
    model_dir = "segmentation_models_pytorch/experiments/"
    model_name = "unet"
    # encoder = "resnet34"
    encoder = "resnext50_32x4"

    output_dir = model_dir + model_name + "-" + encoder + "/"

    if os.path.isdir(output_dir):
        print("rmtree before extracting slices:", output_dir)
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    exported_slices_dir_test = input_dir + "/tif_slices_test/"
    test_dataset = MriDataset(
        path=exported_slices_dir_test,
        augmentation=get_test_augmentation(),
        preprocessing=get_preprocessing(),
    )
    print("test_dataset: " + str(len(test_dataset)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(
        model_dir + model_name + "-" + encoder + ".pth",
        map_location=torch.device(device),
    )

    metrics = [
        smp.utils.metrics.IoU(eps=1.0),
        smp.utils.metrics.Fscore(eps=1.0),
    ]
    metric_iou = []
    metric_dice = []
    image_volume = []
    gt_volume = []
    pr_volume = []
    print("time start", get_datetime_str())

    # axis 0/1 [130: 415]
    # axis 2 [all]
    # for idx in range(150, len(test_dataset) - 200, 50):  # test
    for idx in range(0, len(test_dataset), 1):
        # for idx in range(130, len(test_dataset) - 100, 1):
        image, gt_mask = test_dataset[idx]
        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        gt_mask = torch.from_numpy(gt_mask)

        image_volume.append(image)
        gt_volume.append(gt_mask)
        pr_volume.append(pr_mask)

        iou = metrics[0](pr_mask, gt_mask).cpu().detach().numpy()
        dice = metrics[1](pr_mask, gt_mask).cpu().detach().numpy()
        metric_iou.append(iou)
        metric_dice.append(dice)

        gt_mask = gt_mask.squeeze()
        pr_mask = pr_mask.squeeze().cpu().numpy()
        pr_mask = pr_mask.round()  # use threshold?

        smp.utils.custom_functions.plot_masks_overlay(
            output_path=output_dir + "/masks-" + str(idx) + ".png",
            image=image[0],
            gt_mask=gt_mask,
            pr_mask=pr_mask,
        )
        # smp.utils.custom_functions.visualize(
        #     output_path=output_dir + "/split-" + str(idx) + ".png",
        #     gt_mask=gt_mask,
        #     pr_mask=pr_mask,
        # )
    print("time end", get_datetime_str())
    plot_metrics_per_slice(metric_dice, metric_iou, output_dir)


def save_sample_image(image, mask, output_file):
    fig = plt.figure(figsize=(14, 6))
    rows, columns = 1, 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(image, cmap="gray")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(mask, cmap="gray")
    # plt.show()
    plt.savefig(fname=output_file)
    plt.close(fig)


def save_tif_to_png(
    image, mask=None, output_dir=None, new_fn="input_sample.png",
):
    fig = plt.figure(figsize=(4, 4))
    rows, columns = 1, 1
    fig.add_subplot(rows, columns, 1)
    plt.imshow(image, cmap="gray")
    if mask:
        fig.add_subplot(rows, columns, 2)
        plt.imshow(mask)
    # plt.show()
    plt.savefig(fname=output_dir + new_fn)
    plt.close(fig)


def inference_arg_parser():
    parser = argparse.ArgumentParser(
        description="split 3d image into multiple 2d images"
    )
    parser.add_argument(
        "-in",
        "--input_dir",
        type=str,
        default="/home/a/Thesis/datasets/mri/final_dataset/",
        help="path to NRRD image/volume directory",
    )
    return parser


def get_slices_paths(input_dir: str):
    """Get paths to volumes

    :param input_dir: directory with MRI volumes
    :return: sorted list of paths to masks and volume images
    """
    fns = []
    mask_fns = []
    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if "seg" in filename:
                mask_fns.append(os.path.join(dirname, filename))
            else:
                fns.append(os.path.join(dirname, filename))
    fns = sorted(fns)
    mask_fns = sorted(mask_fns)
    return fns, mask_fns


def convert_tif_folder_to_png(input_dir):
    output_dir = input_dir + "/png"
    if os.path.isdir(output_dir):
        print("rmtree before extracting slices:", output_dir)
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print(output_dir)

    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if "seg" not in filename:
                image = read_pil_image(input_dir + "/" + filename)
                save_tif_to_png(
                    image.T,
                    output_dir=output_dir + "/",
                    new_fn=str(filename).split(".")[0] + ".png",
                )


if __name__ == "__main__":
    # run_inference()
    input_dirs = [
        "/home/a/Thesis/datasets/mri/datasets_n4/tif_slices_train",
        # '/home/a/Thesis/datasets/mri/datasets_n4/tif_slices_test',
        # '/home/a/Thesis/datasets/mri/datasets_n4/tif_slices_valid',
    ]
    for input_dir in input_dirs:
        convert_tif_folder_to_png(input_dir)
