import copy
import math
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from functools import partial
import torch.nn.functional as F

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights

from torchvision.ops.misc import Conv2dNormActivation
from torchvision.models.efficientnet import _MBConvConfig, MBConvConfig, FusedMBConvConfig, _efficientnet_conf
from torchvision.models.efficientnet import EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights

class CustomResNet(nn.Module):
    def __init__(
        self,
        layers: List[int],
        block=BasicBlock,
        zero_init_residual=False,
        groups=1,
        num_classes=1000,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):

        super(CustomResNet, self).__init__()

        self.dict_encoder_features = {}

        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        self.dict_encoder_features["block_1"] = x

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def _resnet(block_type, layers, weights=None, progress=True):
    model = CustomResNet(layers, block_type)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def resnet18(pretrained=True):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None
    return _resnet(BasicBlock, [2, 2, 2, 2], weights=weights)

def resnet34(pretrained=True):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        weights = ResNet34_Weights.IMAGENET1K_V1
    else:
        weights = None
    return _resnet(BasicBlock, [3, 4, 6, 3], weights=weights)

def resnet50(pretrained=True):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V1
    else:
        weights = None
    return _resnet(Bottleneck, [3, 4, 6, 3], weights=weights)


def resnet101(pretrained=True):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        weights = ResNet101_Weights.IMAGENET1K_V1
    else:
        weights = None
    return _resnet(Bottleneck, [3, 4, 23, 3], weights=weights)


class CustomEfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        EfficientNet V1 and V2 main class
        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()
        self.dict_encoder_features = {}

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if "block" in kwargs:
            warnings.warn(
                "The parameter 'block' is deprecated since 0.13 and will be removed 0.15. "
                "Please pass this information on 'MBConvConfig.block' instead."
            )
            if kwargs["block"] is not None:
                for s in inverted_residual_setting:
                    if isinstance(s, MBConvConfig):
                        s.block = kwargs["block"]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.num_channels_final_block = lastconv_output_channels

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        return x

def _efficientnet(
    inverted_residual_setting,
    dropout: float,
    last_channel,
    weights=None,
    norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
    progress=True,
    **kwargs: Any,
    ):
    model = CustomEfficientNet(
        inverted_residual_setting,
        dropout,
        last_channel=last_channel,
        norm_layer=norm_layer,
        **kwargs
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def efficientnet_v2_s(pretrained=True, **kwargs: Any):
    which_efficientnet = "efficientnet_v2_s"
    inverted_residual_setting, last_channel = _efficientnet_conf(which_efficientnet)
    if pretrained:
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    else:
        weights = None
    return _efficientnet(
        inverted_residual_setting,
        0.2,
        last_channel,
        weights=weights,
        **kwargs,
    )

def efficientnet_v2_m(pretrained=True, **kwargs: Any):
    which_efficientnet = "efficientnet_v2_m"
    inverted_residual_setting, last_channel = _efficientnet_conf(which_efficientnet)
    if pretrained:
        weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
        weights = EfficientNet_V2_S_Weights.verify(weights)
    else:
        weights = None
    return _efficientnet(
        inverted_residual_setting,
        0.3,
        last_channel,
        weights=weights,
        **kwargs,
    )

def efficientnet_v2_l(pretrained=True, **kwargs: Any):
    which_efficientnet = "efficientnet_v2_l"
    inverted_residual_setting, last_channel = _efficientnet_conf(which_efficientnet)
    if pretrained:
        weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
    else:
        weights = None
    return _efficientnet(
        inverted_residual_setting,
        0.4,
        last_channel,
        weights=weights,
        **kwargs,
    )
