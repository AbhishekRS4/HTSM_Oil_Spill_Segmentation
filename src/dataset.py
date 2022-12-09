import os
import json
import torch
import numpy as np
from PIL import Image
from skimage.io import imread
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class M4DSAROilSpillDataset(Dataset):
    def __init__(self, dir_data, list_images, which_set="train", file_stats_json="image_stats.json"):
        self.dir_data = dir_data
        self.which_set = which_set
        self.file_stats_json = file_stats_json
        self.dir_images = os.path.join(self.dir_data, "images")
        self.dir_labels = os.path.join(self.dir_data, "labels_1D")

        self.list_images = sorted(list_images)
        self.list_labels = [f.replace(".jpg", ".png") for f in self.list_images]
        self.dict_stats = None

        try:
            with open(file_stats_json) as fh:
                self.dict_stats = json.load(fh)
        except:
            print(f"{self.file_stats_json} not found")

        if self.which_set == "train":
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Pad((15, 11), fill=0),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
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
        else:
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Pad((15, 11), fill=0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        file_image = os.path.join(self.dir_images, self.list_images[idx])
        file_label = os.path.join(self.dir_labels, self.list_labels[idx])

        image = imread(file_image)
        label = imread(file_label)
        label_dims = label.shape

        image = self.image_transform(image)
        label = torch.nn.functional.pad(
            torch.from_numpy(label), (15, 15, 11, 11), value=0
        )

        return image, label

def get_dataloaders_for_training(dir_dataset, batch_size, num_workers=4):
    list_images = sorted(
        [f for f in os.listdir(os.path.join(dir_dataset, "train", "images")) if f.endswith(".jpg")]
    )
    list_train_images, list_valid_images = train_test_split(
        list_images, test_size=0.05, shuffle=True
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
