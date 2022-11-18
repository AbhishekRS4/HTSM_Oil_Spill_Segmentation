import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from decoder_models import DeepLabV3Plus
from encoder_models import resnet18, resnet34, resnet50, resnet101

class ResNet18DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        self.encoder = resnet18(pretrained=pretrained)
        self.segmenter = DeepLabV3Plus(in_channels=512, encoder_channels=64, num_classes=num_classes)

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(encoded_features, self.encoder.dict_encoder_features["block_1"])
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x

class ResNet34DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        self.encoder = resnet34(pretrained=pretrained)
        self.segmenter = DeepLabV3Plus(in_channels=512, encoder_channels=64, num_classes=num_classes)

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(encoded_features, self.encoder.dict_encoder_features["block_1"])
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x

class ResNet50DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        self.encoder = resnet50(pretrained=pretrained)
        self.segmenter = DeepLabV3Plus(in_channels=2048, encoder_channels=64, num_classes=num_classes)

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(encoded_features, self.encoder.dict_encoder_features["block_1"])
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x

class ResNet101DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        self.encoder = resnet101(pretrained=pretrained)
        self.segmenter = DeepLabV3Plus(in_channels=2048, encoder_channels=64, num_classes=num_classes)

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(encoded_features, self.encoder.dict_encoder_features["block_1"])
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x
