import gc
import os
import sys
import traceback
import logging
import warnings
import adabound
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.experiments.helpers import get_model_by_name
from segmentation_models_pytorch.experiments.helpers import arg_parser
from segmentation_models_pytorch.experiments.helpers import combine_results
from segmentation_models_pytorch.experiments.helpers import format_history
from segmentation_models_pytorch.experiments.helpers import format_metrics
from segmentation_models_pytorch.experiments.helpers import get_best_metrics
from segmentation_models_pytorch.experiments.helpers import get_datetime_str
from segmentation_models_pytorch.experiments.helpers import get_volume_fn_from_mask_fn
from segmentation_models_pytorch.experiments.helpers import get_volume_paths
from segmentation_models_pytorch.experiments.helpers import plot_graphs
from segmentation_models_pytorch.experiments.helpers import plot_metrics_per_slice
from segmentation_models_pytorch.experiments.helpers import save_results
from segmentation_models_pytorch.experiments.helpers import save_sample_image
from segmentation_models_pytorch.experiments.helpers import send_email
from segmentation_models_pytorch.utils.custom_functions import get_preprocessing
from segmentation_models_pytorch.utils.custom_functions import get_test_augmentation
from segmentation_models_pytorch.utils.custom_functions import set_global_seed
from segmentation_models_pytorch.utils.custom_functions import (
    get_train_augmentation_medium,
)
from segmentation_models_pytorch.utils.data import MriDataset

plt.rcParams["figure.figsize"] = (7, 7)
warnings.filterwarnings("ignore")

logger = None


def new_print(*args):
    global logger
    return logger.info(" ".join(str(a) for a in args))


def train_model(
    model_name: str,
    encoder: str,
    input_dir: dict,
    output_dir: str,
    batch_size: int,
    epochs: int = 1,
):
    """Script to train networks on MRI dataset

    :param model_name: one of unet/pspnet/fpn/linknet/fcn
    :param encoder: see encoders list at https://github.com/utegulovalmat/segmentation_models.pytorch
    :param input_dir: dict with paths to folders with volumes and masks
    :param output_dir: output folder for model predictions
    :param batch_size: batch size
    :param epochs: number of epochs to train
    :return: (str, dict) results message and dict with metrics
    """
    global logger
    logger.info("input_dir: " + str(input_dir))
    exported_slices_dir_train = input_dir["train"]
    exported_slices_dir_valid = input_dir["valid"]
    exported_slices_dir_test = input_dir["test"]

    path, dirs, files = next(os.walk(exported_slices_dir_train))
    logger.info("exported_slices_dir_train: " + str(len(files) / 2))
    path, dirs, files = next(os.walk(exported_slices_dir_valid))
    logger.info("exported_slices_dir_valid: " + str(len(files) / 2))
    path, dirs, files = next(os.walk(exported_slices_dir_test))
    logger.info("exported_slices_dir_test: " + str(len(files) / 2))

    # Define datasets
    # augmentation = get_train_augmentation_low()
    # augmentation = get_train_augmentation_medium()
    # augmentation = get_train_augmentation_hardcore()
    train_dataset = MriDataset(
        path=exported_slices_dir_train,
        augmentation=get_train_augmentation_medium(),
        preprocessing=get_preprocessing(),
    )
    logger.info("test_dataset: " + str(len(train_dataset)))
    valid_dataset = MriDataset(
        path=exported_slices_dir_valid,
        augmentation=get_test_augmentation(),
        preprocessing=get_preprocessing(),
    )
    logger.info("valid_dataset: " + str(len(valid_dataset)))
    test_dataset = MriDataset(
        path=exported_slices_dir_test,
        augmentation=get_test_augmentation(),
        preprocessing=get_preprocessing(),
    )
    logger.info("test_dataset: " + str(len(test_dataset)))

    image, mask = train_dataset[150]
    new_print("Image and mask dimensions")
    new_print(type(image), image.shape, mask.shape)

    # Show sample image
    save_sample_image(image[0], mask[0], output_dir)

    # Create segmentation model with pretrained encoder
    classes = ["1"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Use device: " + device)
    activation = "sigmoid" if len(classes) == 1 else "softmax"
    logger.info("Activation: " + activation)
    model = get_model_by_name(
        model_name,
        encoder,
        encoder_weights="imagenet",
        classes=classes,
        activation=activation,
    )

    # Note: this will init kernels with random values
    # for m in model.modules():
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)

    # Define metrics, loss and optimizer
    metrics = [
        smp.utils.metrics.IoU(eps=1.0),
        smp.utils.metrics.Fscore(eps=1.0),
    ]
    loss = smp.utils.losses.DiceLoss(eps=1.0)
    # https://www.luolc.com/publications/adabound/
    # optimizer = adabound.AdaBound(model.parameters(), lr=1e-4, final_lr=1e-5)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4
    )  # weight_decay=0.001, amsgrad=True

    # Create DataLoaders
    subset_sampler = SubsetRandomSampler(indices=[150, 160])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=12,
        shuffle=True,
        # sampler=subset_sampler,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        # sampler=subset_sampler,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        # sampler=subset_sampler,
    )
    # Create epoch runners, it is a simple loop of iterating over DataLoader's samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model, loss=loss, metrics=metrics, device=device, verbose=True,
    )

    best_model_fn = "/" + model_name + "-" + encoder + ".pth"
    max_score = 0
    best_epoch = 0
    train_history = []
    valid_history = []
    early_stop_epochs = 0
    for epoch in range(0, epochs):
        new_print("\nEpoch: {}".format(epoch))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_history.append(train_logs)
        valid_history.append(valid_logs)

        if max_score < valid_logs["fscore"]:
            max_score = valid_logs["fscore"]
            torch.save(model, output_dir + best_model_fn)
            logger.info("Model saved at epoch: " + str(epoch))
            best_epoch = epoch
            early_stop_epochs = 0
        else:
            early_stop_epochs += 1
            logger.info("Early stop epochs = " + str(early_stop_epochs))
            if early_stop_epochs == 5:
                optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] / 3
                logger.info(
                    "Decrease learning rate to " + str(optimizer.param_groups[0]["lr"])
                )
            if early_stop_epochs == 7:
                logger.info("Early stopping at epoch: " + str(epoch))
                break

    history = format_history(train_history, valid_history)
    train_metrics = get_best_metrics(history=history, mode="train")
    valid_metrics = get_best_metrics(history=history, mode="val")
    best_train_row = format_metrics(metrics=train_metrics, mode="train")
    best_valid_row = format_metrics(metrics=valid_metrics, mode="val")
    plot_graphs(history, output_dir)
    logger.info("Train performance metrics")
    logger.info(best_train_row)
    logger.info("Validation performance metrics")
    logger.info(best_valid_row)

    # Evaluate model on test set, load best saved checkpoint
    model = torch.load(output_dir + best_model_fn)
    test_epoch = smp.utils.train.ValidEpoch(
        model, loss=loss, metrics=metrics, device=device, verbose=True
    )
    test_metrics = test_epoch.run(test_loader)
    test_metrics = get_best_metrics(history=test_metrics, mode="test")
    best_test_row = format_metrics(test_metrics, mode="test")
    logger.info("Test performance metrics")
    logger.info(best_test_row)

    # Visualize predictions
    metric_iou = []
    metric_dice = []
    image_volume = []
    gt_volume = []
    pr_volume = []
    # for idx in range(150, len(test_dataset) - 200, 50):
    for idx in range(130, len(test_dataset) - 100, 1):
        image, gt_mask = test_dataset[idx]
        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        pr_mask = model.predict(x_tensor).to(device)
        gt_mask = torch.from_numpy(gt_mask).to(device)

        image_volume.append(image)
        gt_volume.append(gt_mask)
        pr_volume.append(pr_mask)

        iou = metrics[0](pr_mask, gt_mask).cpu().detach().numpy()
        dice = metrics[1](pr_mask, gt_mask).cpu().detach().numpy()
        metric_iou.append(iou)
        metric_dice.append(dice)

        gt_mask = gt_mask.squeeze().cpu().numpy()
        pr_mask = pr_mask.squeeze().cpu().numpy()
        pr_mask = pr_mask.round()  # use threshold?

        smp.utils.custom_functions.plot_masks_overlay(
            output_path=output_dir + "/masks-" + str(idx) + ".png",
            image=image[0],
            gt_mask=gt_mask,
            pr_mask=pr_mask,
        )

        # smp.utils.custom_functions.visualize(
        #     output_path=output_dir + "/" + str(idx) + ".png",
        #     gt_mask=gt_mask,
        #     pr_mask=pr_mask,
        # )

    # count metrics per slice
    plot_metrics_per_slice(metric_dice, metric_iou, output_dir)

    results = combine_results(train_metrics, valid_metrics, test_metrics)
    message = "Model: <strong>" + model_name + "-" + encoder + "</strong><br>"
    message += "Best valid epoch: " + str(best_epoch) + "<br>"
    message += "Train<br>\n" + best_train_row + "\n<br>"
    message += "Valid<br>\n" + best_valid_row + "\n<br>"
    message += "Test<br>\n" + best_test_row + "\n<br>"
    return message, results


def main():
    """
    source ~/ml-env3/bin/activate
    python -m segmentation_models_pytorch.experiments.train_model -in /home/a/Thesis/datasets/mri/final_dataset --train_all all --extract_slices 1 --use_axis 1

    # train using axis 0 - all volumes
    nohup python -m segmentation_models_pytorch.experiments.train_model -in /datastore/home/segnet/datasets --train_all all --extract_slices 1 --use_axis 0 &

    # train using axis 1 - all volumes
    nohup python -m segmentation_models_pytorch.experiments.train_model -in /datastore/home/segnet/datasets --train_all all --extract_slices 1 --use_axis 1 &

    # train using axis 2 - all volumes
    nohup python -m segmentation_models_pytorch.experiments.train_model -in /datastore/home/segnet/datasets --train_all all --extract_slices 1 --use_axis 2 &

    # train using axis 012 - all volumes
    nohup python -m segmentation_models_pytorch.experiments.train_model -in /datastore/home/segnet/datasets --train_all all --extract_slices 1 --use_axis 012 &

    >> [<pid>] # returns process id
    echo pid >> last_pid.txt
    tail nohup.out -f
    """
    global logger
    base_path = "segmentation_models_pytorch/experiments/"
    results_file = "segmentation_models_pytorch/experiments/results.csv"
    pipline_file = "segmentation_models_pytorch/experiments/pipeline.csv"
    pipeline = pd.read_csv(pipline_file, dtype={"axis": str, "epochs": int},)
    set_global_seed(42)

    # Get arguments
    args = arg_parser().parse_args()
    input_dir = args.input_dir
    extract_slices = args.extract_slices
    train_all = args.train_all == "all"
    use_axis = args.use_axis

    # Get paths to volumes and masks
    mask_fns, fns = get_volume_paths(input_dir)
    n_volumes = 12 if train_all else 1
    train_masks = mask_fns[0:n_volumes]
    train_volumes = [get_volume_fn_from_mask_fn(fn) for fn in train_masks]
    valid_masks = mask_fns[12:13]
    valid_volumes = [get_volume_fn_from_mask_fn(fn) for fn in valid_masks]
    test_masks = mask_fns[13:14]
    test_volumes = [get_volume_fn_from_mask_fn(fn) for fn in test_masks]

    # Extract slices from volumes
    dataset_dir = "/".join(input_dir.split("/")[:-1])
    exported_slices_dir_train = dataset_dir + "/tif_slices_train/"
    exported_slices_dir_valid = dataset_dir + "/tif_slices_valid/"
    exported_slices_dir_test = dataset_dir + "/tif_slices_test/"
    input_dir = {
        "train": exported_slices_dir_train,
        "valid": exported_slices_dir_valid,
        "test": exported_slices_dir_test,
    }
    if extract_slices:
        print(get_datetime_str(), "train", train_volumes, train_masks)
        smp.utils.custom_functions.extract_slices_from_volumes(
            images=train_volumes,
            masks=train_masks,
            output_dir=exported_slices_dir_train,
            skip_empty_mask=True,
            use_dimensions=use_axis,
        )
        print(get_datetime_str(), "valid", valid_volumes, valid_masks)
        smp.utils.custom_functions.extract_slices_from_volumes(
            images=valid_volumes,
            masks=valid_masks,
            output_dir=exported_slices_dir_valid,
            skip_empty_mask=True,
            use_dimensions=use_axis,
        )
        print(get_datetime_str(), "test", test_volumes, test_masks)
        smp.utils.custom_functions.extract_slices_from_volumes(
            images=test_volumes,
            masks=test_masks,
            output_dir=exported_slices_dir_test,
            skip_empty_mask=False,
            use_dimensions=use_axis,
        )

    for idx, (done, model, encoder, _axis, epochs, batch) in pipeline.iterrows():
        if done == "yes":
            continue
        if use_axis != _axis:
            continue
        cur_datetime = get_datetime_str()
        output_dir = base_path + "-".join([model, encoder, use_axis, cur_datetime])
        os.mkdir(output_dir)
        logger = get_logger(output_dir)
        logger.info("done, model, encoder, axis, epochs, batch")
        logger.info(
            " ".join([str(i) for i in [done, model, encoder, use_axis, epochs, batch]])
        )
        new_print("Extract slices:", extract_slices)
        new_print("train", get_fns(train_volumes))
        new_print("valid", get_fns(valid_volumes))
        new_print("test", get_fns(test_volumes))
        try:
            logger.info("\n\n\nStart training " + output_dir + "\n\n")

            # message, results = "", {}
            message, results = train_model(
                model_name=model,
                encoder=encoder,
                output_dir=output_dir,
                epochs=epochs,
                input_dir=input_dir,
                batch_size=batch,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            logger.info("Finish")
            title = model + "-" + encoder + " SUCCESS"
            prefix = "Training finished with status: " + title + "\n\n"
            message = prefix + message
            logger.info(message + "\n\n" + "=" * 100)
            mask = (pipeline["model"] == model) & (pipeline["encoder"] == encoder)
            pipeline["done"][mask] = "yes"
            pipeline.to_csv(pipline_file, index=False)

            model_results = {
                "model": model,
                "encoder": encoder,
                "axis": use_axis,
                "imagenet": "yes",  # TODO: pass this to model trainer
            }
            model_results.update(results)
            last_row = save_results(filename=results_file, results=model_results)
            message += "<br>" + str(last_row.to_list())
            logger.info("Send email")
        except Exception as e:
            logger.error("Exception")
            logger.error(str(e))
            logger.error(traceback.format_exc())
            title = model + "-" + encoder + " FAILED"
            logger.info("Send email")
            message = traceback.format_exc()
        print(message)
        send_email(title=title, message=message)
        # break
    return 0


def get_logger(output_dir):
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_filename = output_dir + "/train.log"
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_fns(x):
    r = []
    for i in x:
        r.append(i.split("/")[-1])
    return r


if __name__ == "__main__":
    """
    Encoders: https://github.com/qubvel/segmentation_models.pytorch

    ## Axis 0
    ls tif_slices_valid | wc -l ## 490
    ls tif_slices_test | wc -l ## 352
    ls tif_slices_train | wc -l ## 5372 - all volumes
    ls tif_slices_train | wc -l ## 454 - one volume

    ## Axis 012
    ls tif_slices_valid/ | wc -l ## 1254
    ls tif_slices_test | wc -l ## 1206
    ls tif_slices_train | wc -l ## 1226 - one volume
    ls tif_slices_train | wc -l ## ... - all volumes
    """
    sys.exit(main())
