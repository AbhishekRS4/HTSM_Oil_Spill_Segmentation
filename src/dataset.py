import os
import json
import torch
import numpy as np
from PIL import Image
from skimage.io import imread
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from image_preprocessing import ImagePadder
from logger_utils import load_dict_from_json

class M4DSAROilSpillDataset(Dataset):
    def __init__(self, dir_data, list_images, which_set="train", file_stats_json="image_stats.json"):
        self.dir_data = dir_data
        self.which_set = which_set
        self.file_stats_json = file_stats_json
        try:
            self.dict_stats = load_dict_from_json(self.file_stats_json)
        except:
            dir_json = os.path.dirname(os.path.realpath(__file__))
            self.dict_stats = load_dict_from_json(os.path.join(dir_json, self.file_stats_json))
        self._dir_images = os.path.join(self.dir_data, "images")
        self._dir_labels = os.path.join(self.dir_data, "labels_1D")

        self._list_images = sorted(list_images)
        self._list_label = [f.replace(".jpg", ".png") for f in self._list_images]

        self._image_padder = ImagePadder(self._dir_images)
        self._affine_transform = None

        self._image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    self.dict_stats["mean"],
                    self.dict_stats["mean"],
                    self.dict_stats["mean"]
                ],
                std=[
                    self.dict_stats["std"],
                    self.dict_stats["std"],
                    self.dict_stats["std"]
                ]
            ),
        ])

        if self.which_set == "train":
            self._affine_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])


    def __len__(self):
        return len(self._list_images)

    def __getitem__(self, idx):
        file_image = os.path.join(self._dir_images, self._list_images[idx])
        file_label = os.path.join(self._dir_labels, self._list_label[idx])

        image = imread(file_image)
        label = imread(file_label)

        image = self._image_padder.pad_image(image)
        label = self._image_padder.pad_label(label)

        if self.which_set == "train":
            image_tensor = torch.from_numpy(image)
            # H x W x 3
            label_tensor = torch.from_numpy(label)
            # H x W
            label_tensor = torch.unsqueeze(label_tensor, dim=-1)
            # H x W x 1
            stacked = torch.cat([image_tensor, label_tensor], dim=-1)
            # H x W x 4
            stacked = torch.permute(stacked, [2, 0, 1])
            # 4 x H x W
            stacked_transformed = self._affine_transform(stacked)
            # 4 x H x W
            stacked_transformed = torch.permute(stacked_transformed, [1, 2, 0])
            # H x W x 4
            stacked_arr = stacked_transformed.numpy()

            image = stacked_arr[:, :, :-1]
            # H x W x 3
            label = stacked_arr[:, :, -1]
            # H x W

        image = self._image_transform(image)
        return image, label

def get_dataloaders_for_training(dir_dataset, batch_size, random_state=None, num_workers=4):
    list_images = sorted(
        [f for f in os.listdir(os.path.join(dir_dataset, "train", "images")) if f.endswith(".jpg")]
    )
    list_train_images, list_valid_images = train_test_split(
        list_images, test_size=0.05, shuffle=True, random_state=random_state,
    )
    print("dataset information")
    print(f"number of train samples: {len(list_train_images)}")
    print(f"number of validation samples: {len(list_valid_images)}")

    train_dataset = M4DSAROilSpillDataset(
        os.path.join(dir_dataset, "train"),
        list_train_images,
        which_set="train"
    )
    train_dataset_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    valid_dataset = M4DSAROilSpillDataset(
        os.path.join(dir_dataset, "train"),
        list_valid_images,
        which_set="valid"
    )
    valid_dataset_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_dataset_loader, valid_dataset_loader
