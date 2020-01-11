import argparse
import base64
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.convnet.model import ConvNet

plt.rcParams["figure.figsize"] = (7, 7)

logger = None


class NoMatchingModelException(Exception):
    pass


def get_volume_paths(input_dir: str):
    """Get paths to volumes

    :param input_dir: directory with MRI volumes
    :return: sorted list of paths to masks and volume images
    """
    fns = []
    mask_fns = []
    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if "nrrd" in filename and "seg" not in filename:
                if "label.nrrd" in filename or "-label-corr.nrrd" in filename:
                    mask_fns.append(os.path.join(dirname, filename))
                else:
                    fns.append(os.path.join(dirname, filename))
    mask_fns = sorted(mask_fns)
    fns = sorted(fns)
    return mask_fns, fns


def get_volume_fn_from_mask_fn(mask_filename):
    """Extract volume path from mask path"""
    return mask_filename.replace("S-label", "")


def save_sample_image(image, mask, output_dir):
    fig = plt.figure(figsize=(7, 7))
    rows, columns = 1, 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(image)
    fig.add_subplot(rows, columns, 2)
    plt.imshow(mask)
    # plt.show()
    plt.savefig(fname=output_dir + "/input_sample.png")


def format_history(train_history, valid_history):
    history = {
        "loss": [],
        "iou_score": [],
        "f-score": [],
        "val_loss": [],
        "val_iou_score": [],
        "val_f-score": [],
    }
    for train_log, valid_log in zip(train_history, valid_history):
        history["loss"].append(train_log["dice_loss"])
        history["iou_score"].append(train_log["iou_score"])
        history["f-score"].append(train_log["fscore"])
        history["val_loss"].append(valid_log["dice_loss"])
        history["val_iou_score"].append(valid_log["iou_score"])
        history["val_f-score"].append(valid_log["fscore"])
    return history


def get_best_metrics(history, mode):
    if mode == "train":
        best_train_loss = 1e10
        train_metrics = None
        for idx, (loss, iou, dice) in enumerate(
            zip(history["loss"], history["iou_score"], history["f-score"])
        ):
            if loss < best_train_loss:
                best_train_loss = loss
                train_metrics = {
                    "epoch": str(idx),
                    "loss": best_train_loss,
                    "iou": iou,
                    "dice": dice,
                }
        print("Best train |", train_metrics)
        return train_metrics

    if mode == "val":
        best_valid_loss = 1e10
        valid_metrics = None
        for idx, (loss, iou, dice) in enumerate(
            zip(history["val_loss"], history["val_iou_score"], history["val_f-score"])
        ):
            if loss < best_valid_loss:
                best_valid_loss = loss
                valid_metrics = {
                    "epoch": str(idx),
                    "loss": best_valid_loss,
                    "iou": iou,
                    "dice": dice,
                }
        print("Best valid |", valid_metrics)
        return valid_metrics

    if mode == "test":
        test_metrics = {
            "epoch": 0,
            "loss": history["dice_loss"],
            "iou": history["iou_score"],
            "dice": history["fscore"],
        }
        return test_metrics


def format_metrics(metrics, mode):
    if mode == "train":
        epoch = metrics["epoch"]
        loss = metrics["loss"]
        iou = metrics["iou"]
        dice = metrics["dice"]
    elif mode == "val":
        epoch = metrics["epoch"]
        loss = metrics["loss"]
        iou = metrics["iou"]
        dice = metrics["dice"]
    elif mode == "test":
        epoch = 0
        loss = metrics["loss"]
        iou = metrics["iou"]
        dice = metrics["dice"]
    else:
        raise
    loss = "loss: \n{:.5}".format(loss)
    iou = "iou: \n{:.5}".format(iou)
    dice = "dice: \n{:.5}".format(dice)
    return "\n ".join(str(i) for i in [epoch, loss, iou, dice])


def combine_results(train_metrics, valid_metrics, test_metrics):
    return {
        "train_loss": train_metrics["loss"],
        "train_iou": train_metrics["iou"],
        "train_dice": train_metrics["dice"],
        "valid_loss": valid_metrics["loss"],
        "valid_iou": valid_metrics["iou"],
        "valid_dice": valid_metrics["dice"],
        "test_loss": test_metrics["loss"],
        "test_iou": test_metrics["iou"],
        "test_dice": test_metrics["dice"],
    }


def plot_metrics_per_slice(dice, iou, output_dir):
    # import seaborn as sns
    # dice = list(zip([i for i in range(0, len(dice))], dice))
    # sns.barplot(x="Slice index", y="DSC, %", data=dice)

    plt.figure(figsize=(10, 5))
    plt.bar(x=[i for i in range(0, len(dice))], height=dice, width=0.3)
    plt.title("DSC per slice")
    plt.ylabel("%")
    plt.xlabel("number of slice")
    plt.savefig(output_dir + "/dice_per_slice.png")
    # plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x=[i for i in range(0, len(iou))], height=iou, width=0.3)
    plt.title("IoU per slice")
    plt.ylabel("%")
    plt.xlabel("number of slice")
    plt.savefig(output_dir + "/iou_per_slice.png")
    plt.close()


def plot_graphs(history, output_dir):
    print("plot_graphs", history["iou_score"])
    print("plot_graphs", history["val_iou_score"])
    # Plot training & validation iou_score values
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.plot(history["iou_score"])
    plt.plot(history["val_iou_score"])
    plt.title("Model IoU score")
    plt.ylabel("IoU score")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")

    # Plot training & validation dice values
    plt.subplot(132)
    plt.plot(history["f-score"])
    plt.plot(history["val_f-score"])
    plt.title("Model F-score")
    plt.ylabel("F-score")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")

    # Plot training & validation loss values
    plt.subplot(133)
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    # plt.show()
    plt.savefig(output_dir + "/graphs.png")


def get_datetime_str():
    from datetime import datetime

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H-%M-%S")
    return date + "-" + time


def arg_parser():
    parser = argparse.ArgumentParser(
        description="split 3d image into multiple 2d images"
    )
    parser.add_argument(
        "-in",
        "--input_dir",
        type=str,
        default="/home/a/Thesis/datasets/mri/final_dataset",
        help="path to NRRD image/volume directory",
    )
    parser.add_argument(
        "-t",
        "--train_all",
        type=str,
        default="one",
        help="use 'all' to train model on 12 volumes, else it will use 1 volume",
    )
    parser.add_argument(
        "--extract_slices",
        type=int,
        default=1,
        help="1 - extract slices, 0 - skip this step",
    )
    parser.add_argument(
        "--use_axis",
        type=str,
        default="012",
        help="which axises to extract for training",
    )
    return parser


def save_results(filename, results):
    df = pd.read_csv(
        filename,
        dtype={
            "model": str,
            "encoder": str,
            "axis": str,
            "imagenet": str,
            "train_loss": np.float32,
            "train_iou": np.float32,
            "train_dice": np.float32,
            "valid_loss": np.float32,
            "valid_iou": np.float32,
            "valid_dice": np.float32,
            "test_loss": np.float32,
            "test_iou": np.float32,
            "test_dice": np.float32,
        },
    )
    df = df.append(other=results, ignore_index=True)
    df.to_csv(filename, index=False)
    return df.iloc[-1]


def send_email(title, message):
    """
    import base64
    encoded = base64.b64encode(b'...')
    data = base64.b64decode(encoded)
    """
    api = os.environ.get("SENDGRID_API_KEY")
    to = base64.b64decode(b"QWxtYXQgPHV0ZWd1bG92QHVuaS1rb2JsZW56LmRlPg==").decode()
    message = Mail(
        from_email=to,
        to_emails=to,
        subject="Model training: " + title,
        html_content=message,
    )
    try:
        sg = SendGridAPIClient(api)
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(str(e))


def get_model_by_name(
    model_name, encoder, encoder_weights=None, classes=None, activation=None
):
    if model_name == "unet":
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=len(classes),
            activation=activation,
            in_channels=1,
        )
    elif model_name == "fpn":
        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=len(classes),
            activation=activation,
            in_channels=1,
        )
    elif model_name == "linknet":
        model = smp.Linknet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=len(classes),
            activation=activation,
            in_channels=1,
        )
    elif model_name == "pspnet":
        model = smp.PSPNet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=len(classes),
            activation=activation,
            in_channels=1,
        )
    elif model_name == "fcn":
        # TODO: add flexibility with encoder selection
        model = smp.FCN(encoder_name=encoder, classes=len(classes),)
    elif model_name == "convnet":
        size = encoder  # convnet size large/small + 32/64
        model = ConvNet(size=size, classes=len(classes),)
    else:
        raise NoMatchingModelException
    return model
