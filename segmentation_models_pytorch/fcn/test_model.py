"""
Run script with:

cd ~/Downloads/Theses/Repos/segmentation_models.pytorch
source ~/ml-env3/bin/activate
python -m segmentation_models_pytorch.fcn.test_model
"""
import copy
from collections import defaultdict

import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from segmentation_models_pytorch.utils.losses import DiceLoss
from .model import FCN
from .simulation import generate_random_data
from .utils import dice_loss
from .utils import masks_to_colorimg
from .utils import plot_side_by_side
from .utils import print_metrics


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics["bce"] += bce.data.cpu().numpy() * target.size(0)
    metrics["dice"] += dice.data.cpu().numpy() * target.size(0)
    metrics["loss"] += loss.data.cpu().numpy() * target.size(0)

    return loss


class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = generate_random_data(
            192, 192, count=count
        )
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]


def main():
    # Generate some random images
    input_images, target_masks = generate_random_data(192, 192, count=1)
    for x in [input_images, target_masks]:
        print(x.shape)
        print(x.min(), x.max())

    # Sanity check for generated images
    # input_images_rgb = [x.swapaxes(0, 2).swapaxes(0, 1).astype(np.uint8) for x in input_images]
    # target_masks_rgb = [masks_to_colorimg(x) for x in target_masks]
    # plot_side_by_side([input_images_rgb, target_masks_rgb])

    # Init model
    device = torch.device("cpu")
    # Unet/FCN: https://github.com/usuyama/pytorch-unet
    model = FCN(classes=6, encoder_name="resnet18",).to(device)
    # print(model.fc1)

    # Dataloaders
    trans = transforms.Compose([transforms.ToTensor(),])
    train_set = SimDataset(2000, transform=trans)
    val_set = SimDataset(100, transform=trans)

    batch_size = 5
    dataloaders = {
        "train": DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=4
        ),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4),
    }

    # Train model
    # criterion = nn.BCELoss()
    criterion = DiceLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    # best_model = train_model(model, criterion, optimizer, num_epochs=20)
    best_model = train_model_v2(model, optimizer, dataloaders, num_epochs=1)

    # View prediction
    images, labels = generate_random_data(192, 192, count=1, reverse_order=True)
    images, labels = torch.from_numpy(images).float(), torch.from_numpy(labels).float()
    # preds = best_model.forward(images).detach().numpy()
    preds = best_model.predict(images)
    metrics = defaultdict(float)
    loss = calc_loss(preds, labels, metrics)
    print("loss", loss)
    print_metrics(metrics, 1, "test")
    preds = preds.detach().numpy()
    images = images.numpy()

    print("PREDS SHAPE", type(preds), preds.shape)
    show_preds = preds[0, :]
    # plt.imshow(show_preds)
    # plt.show()

    # Change channel-order and make 3 channels for matplot
    input_images_rgb = [
        (x.swapaxes(0, 2).swapaxes(0, 1)).astype(np.uint8) for x in images
    ]
    # Map each channel (i.e. class) to each color
    target_masks_rgb = [masks_to_colorimg(x) for x in preds]
    # Left: Input image, Right: Target mask
    print(len(input_images_rgb), input_images_rgb[0].shape)
    print(len(target_masks_rgb), target_masks_rgb[0].shape)
    plot_side_by_side([input_images_rgb, target_masks_rgb])


def train_model_v2(model, optimizer, dataloaders, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    device = torch.device("cpu")

    try:
        load_model = torch.load("./best_model.pth")
        print("load model")
        model = load_model
        # return model
    except:
        print("No model to load")

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                for param_group in optimizer.param_groups:
                    print("LR", param_group["lr"])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics["loss"] / epoch_samples

            # deep copy the model
            if phase == "val" and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print("{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    print("Best val loss: {:4f}".format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_model(model, criterion, optimizer, num_epochs=20, batch_size=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    try:
        load_model = torch.load("./best_model.pth")
        print("load model")
        model = load_model
        # return model
    except:
        print("No model to load")

    # Freeze backbone layers
    # for l in model.base_layers:
    #    for param in l.parameters():
    #        param.requires_grad = False

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        device = torch.device("cpu")
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                epoch_steps = 1
            else:
                model.eval()
                epoch_steps = 1

            running_loss = 0.0
            # Iterate over data.
            # Number of train/test samples = Batch size * epoch_steps
            for i in range(epoch_steps):
                input_images, target_masks = generate_random_data(
                    192, 192, count=batch_size
                )
                # print(input_images.shape, target_masks.shape)
                inputs = torch.from_numpy(input_images).float()
                labels = torch.from_numpy(target_masks).float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    # print("outputs shape", outputs.shape)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / (batch_size * epoch_steps)

            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            # deep copy the model
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val loss: {:4f}".format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model, "./best_model.pth")
    return model


if __name__ == "__main__":
    main()
