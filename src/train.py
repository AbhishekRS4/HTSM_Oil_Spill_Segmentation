import os
import sys
import time
import math
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from metrics import *
from seg_models import *
from dataset import get_dataloaders_for_training
from logger_utils import CSVWriter, write_dict_to_json

from torch.optim.lr_scheduler import _LRScheduler

class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_epochs, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_epochs = max_epochs
        self.min_lr = min_lr # avoid zero lr
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_epochs )**self.power, self.min_lr)
                for base_lr in self.base_lrs]

def validation_loop(dataset_loader, model, ce_loss, device):
    model.eval()
    size = len(dataset_loader.dataset)
    num_batches = len(dataset_loader)
    valid_loss, valid_acc, valid_IOU = 0, 0, 0

    with torch.no_grad():
        for image, label in dataset_loader:
            image = image.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)

            pred_logits = model(image)
            valid_loss += ce_loss(pred_logits, label)

            pred_probs = F.softmax(pred_logits, dim=1)
            pred_label = torch.argmax(pred_probs, dim=1)

            valid_acc += compute_mean_pixel_acc(label, pred_label)
            valid_IOU += compute_mean_IOU(label, pred_label)

    valid_loss /= num_batches
    valid_acc /= num_batches
    valid_IOU /= num_batches
    return valid_loss, valid_acc, valid_IOU

def train_loop(dataset_loader, model, ce_loss, optimizer, device):
    model.train()
    size = len(dataset_loader.dataset)
    num_batches = len(dataset_loader)
    train_loss = 0

    for image, label in dataset_loader:
        image = image.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        optimizer.zero_grad()

        pred_logits = model(image)
        loss = ce_loss(pred_logits, label)

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss
    train_loss /= num_batches
    return train_loss

def batch_train(FLAGS):
    dir_path = os.path.join(FLAGS.dir_model, FLAGS.which_model)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        print(f"created directory : {dir_path}")
    csv_writer = CSVWriter(file_name=os.path.join(dir_path, "train_metrics.csv"),
        column_names=["epoch", "train_loss", "valid_loss", "valid_acc", "valid_IOU"])

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset_loader, valid_dataset_loader = get_dataloaders_for_training(
        FLAGS.dir_dataset, FLAGS.batch_size, random_state=FLAGS.random_state,
    )

    if FLAGS.which_model == "resnet_18_deeplab_v3+":
        oil_spill_seg_model = ResNet18DeepLabV3Plus(
            num_classes=FLAGS.num_classes, pretrained=bool(FLAGS.pretrained)
        )
    elif FLAGS.which_model == "resnet_34_deeplab_v3+":
        oil_spill_seg_model = ResNet34DeepLabV3Plus(
            num_classes=FLAGS.num_classes, pretrained=bool(FLAGS.pretrained)
        )
    elif FLAGS.which_model == "resnet_50_deeplab_v3+":
        oil_spill_seg_model = ResNet50DeepLabV3Plus(
            num_classes=FLAGS.num_classes, pretrained=bool(FLAGS.pretrained)
        )
    elif FLAGS.which_model == "resnet_101_deeplab_v3+":
        oil_spill_seg_model = ResNet101DeepLabV3Plus(
            num_classes=FLAGS.num_classes, pretrained=bool(FLAGS.pretrained)
        )
    elif FLAGS.which_model == "efficientnet_v2_s_deeplab_v3":
        oil_spill_seg_model = EfficientNetSDeepLabV3(
            num_classes=FLAGS.num_classes, pretrained=bool(FLAGS.pretrained)
        )
    elif FLAGS.which_model == "efficientnet_v2_m_deeplab_v3":
        oil_spill_seg_model = EfficientNetMDeepLabV3(
            num_classes=FLAGS.num_classes, pretrained=bool(FLAGS.pretrained)
        )
    elif FLAGS.which_model == "efficientnet_v2_l_deeplab_v3":
        oil_spill_seg_model = EfficientNetLDeepLabV3(
            num_classes=FLAGS.num_classes, pretrained=bool(FLAGS.pretrained)
        )
    else:
        print("model not yet implemented, so exiting")
        sys.exit(0)
    oil_spill_seg_model.to(device)

    if FLAGS.which_optimizer == "sgd":
        optimizer = torch.optim.SGD(
            oil_spill_seg_model.parameters(),
            lr=FLAGS.learning_rate,
            momentum=0.9,
            weight_decay=FLAGS.weight_decay
        )
        lr_scheduler = PolynomialLR(
            optimizer, FLAGS.num_epochs+1, power=0.9,
        )
    elif FLAGS.which_optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            oil_spill_seg_model.parameters(),
            lr=FLAGS.learning_rate,
            weight_decay=FLAGS.weight_decay,
        )

    ce_loss = torch.nn.CrossEntropyLoss()
    print(f"\ntraining oil spill segmentation model: {FLAGS.which_model}\n")
    write_dict_to_json(os.path.join(dir_path, "params.json"), vars(FLAGS))
    for epoch in range(1, FLAGS.num_epochs+1):
        t_1 = time.time()
        train_loss = train_loop(
            train_dataset_loader, oil_spill_seg_model, ce_loss, optimizer, device
        )
        t_2 = time.time()
        print("-"*100)
        print(f"Epoch : {epoch}/{FLAGS.num_epochs}, time: {(t_2-t_1):.2f} sec., train loss: {train_loss:.5f}")
        valid_loss, valid_acc, valid_IOU = validation_loop(
            valid_dataset_loader, oil_spill_seg_model, ce_loss, device
        )
        print(f"validation loss: {valid_loss:.5f}, validation accuracy: {valid_acc:.5f}, validation IOU: {valid_IOU:.5f}")
        csv_writer.write_row(
            [
                epoch,
                np.around(train_loss.cpu().detach().numpy(), 5),
                np.around(valid_loss.cpu().detach().numpy(), 5),
                round(valid_acc, 5),
                round(valid_IOU, 5),
            ]
        )
        torch.save(oil_spill_seg_model.state_dict(), os.path.join(dir_path, f"oil_spill_seg_{FLAGS.which_model}_{epoch}.pt"))
        if FLAGS.which_optimizer == "sgd":
            lr_scheduler.step()
    print("Training complete!!!!")
    csv_writer.close()
    return

def main():
    dir_dataset = "/home/abhishek/Desktop/RUG/htsm_masterwork/oil-spill-detection-dataset/"
    learning_rate = 1e-2
    weight_decay = 1e-4
    which_optimizer = "sgd"
    num_epochs = 50
    batch_size = 32
    num_classes = 5
    which_model = "resnet_18_deeplab_v3+"
    dir_model = os.getcwd()
    list_model_choices = [
        "resnet_18_deeplab_v3+",
        "resnet_34_deeplab_v3+",
        "resnet_50_deeplab_v3+",
        "resnet_101_deeplab_v3+",
        "efficientnet_v2_s_deeplab_v3",
        "efficientnet_v2_m_deeplab_v3",
        "efficientnet_v2_l_deeplab_v3",
    ]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dir_dataset", default=dir_dataset,
        type=str, help="full directory path to the dataset")

    parser.add_argument("--pretrained", default=1,
        type=int, choices=[0, 1], help="use pretrained encoder (1:True, 0:False)")

    parser.add_argument("--random_state", default=3,
        type=int, help="random state to be used to split dataset into train and validation sets")

    parser.add_argument("--which_optimizer", default=which_optimizer,
        type=str, choices=["sgd", "adamw"], help="optimizer to be used for learning")
    parser.add_argument("--learning_rate", default=learning_rate,
        type=float, help="learning rate (1e-4 for AdamW and 1e-2 for SGD)")
    parser.add_argument("--weight_decay", default=weight_decay,
        type=float, help="weight decay")
    parser.add_argument("--num_epochs", default=num_epochs,
        type=int, help="number of epochs to train")
    parser.add_argument("--batch_size", default=batch_size,
        type=int, help="number of samples in a batch")

    parser.add_argument("--num_classes", default=num_classes,
        type=int, help="number of semantic classes in the dataset")

    parser.add_argument("--which_model", default=which_model,
        type=str, choices=list_model_choices, help="which model to train")
    parser.add_argument("--dir_model", default=dir_model,
        type=str, help="base directory where to save the directory with trained checkpoint model files")

    FLAGS, unparsed = parser.parse_known_args()
    batch_train(FLAGS)
    return

if __name__ == "__main__":
    main()
