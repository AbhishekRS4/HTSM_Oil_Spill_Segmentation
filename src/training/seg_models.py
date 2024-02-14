import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from decoder_models import DeepLabV3Plus, DeepLabV3
from encoder_models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    efficientnet_v2_s,
    efficientnet_v2_m,
    efficientnet_v2_l,
)


class ResNet18DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        ----------
        Attributes
        ----------
        num_classes : int
            number of classes in the dataset
        pretrained : bool
            indicates whether to load pretrained weights for the encoder model (default: True)
        """
        super().__init__()

        self.encoder = resnet18(pretrained=pretrained)
        self.segmenter = DeepLabV3Plus(
            in_channels=512, encoder_channels=64, num_classes=num_classes
        )

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(
            encoded_features, self.encoder.dict_encoder_features["block_1"]
        )
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class ResNet34DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        ----------
        Attributes
        ----------
        num_classes : int
            number of classes in the dataset
        pretrained : bool
            indicates whether to load pretrained weights for the encoder model (default: True)
        """
        super().__init__()

        self.encoder = resnet34(pretrained=pretrained)
        self.segmenter = DeepLabV3Plus(
            in_channels=512, encoder_channels=64, num_classes=num_classes
        )

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(
            encoded_features, self.encoder.dict_encoder_features["block_1"]
        )
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class ResNet50DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        ----------
        Attributes
        ----------
        num_classes : int
            number of classes in the dataset
        pretrained : bool
            indicates whether to load pretrained weights for the encoder model (default: True)
        """
        super().__init__()

        self.encoder = resnet50(pretrained=pretrained)
        self.segmenter = DeepLabV3Plus(
            in_channels=2048, encoder_channels=64, num_classes=num_classes
        )

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(
            encoded_features, self.encoder.dict_encoder_features["block_1"]
        )
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class ResNet101DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        ----------
        Attributes
        ----------
        num_classes : int
            number of classes in the dataset
        pretrained : bool
            indicates whether to load pretrained weights for the encoder model (default: True)
        """
        super().__init__()

        self.encoder = resnet101(pretrained=pretrained)
        self.segmenter = DeepLabV3Plus(
            in_channels=2048, encoder_channels=64, num_classes=num_classes
        )

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(
            encoded_features, self.encoder.dict_encoder_features["block_1"]
        )
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class EfficientNetSDeepLabV3(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        ----------
        Attributes
        ----------
        num_classes : int
            number of classes in the dataset
        pretrained : bool
            indicates whether to load pretrained weights for the encoder model (default: True)
        """
        super().__init__()

        self.encoder = efficientnet_v2_s(pretrained=pretrained)
        self.segmenter = DeepLabV3(
            in_channels=self.encoder.num_channels_final_block,
            num_classes=num_classes,
        )

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(encoded_features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class EfficientNetMDeepLabV3(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        ----------
        Attributes
        ----------
        num_classes : int
            number of classes in the dataset
        pretrained : bool
            indicates whether to load pretrained weights for the encoder model (default: True)
        """
        super().__init__()

        self.encoder = efficientnet_v2_m(pretrained=pretrained)
        self.segmenter = DeepLabV3(
            in_channels=self.encoder.num_channels_final_block,
            num_classes=num_classes,
        )

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(encoded_features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class EfficientNetLDeepLabV3(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        ----------
        Attributes
        ----------
        num_classes : int
            number of classes in the dataset
        pretrained : bool
            indicates whether to load pretrained weights for the encoder model (default: True)
        """
        super().__init__()

        self.encoder = efficientnet_v2_l(pretrained=pretrained)
        self.segmenter = DeepLabV3(
            in_channels=self.encoder.num_channels_final_block,
            num_classes=num_classes,
        )

    def forward(self, x):
        input_shape = x.shape[2:]
        encoded_features = self.encoder(x)
        x = self.segmenter(encoded_features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x
