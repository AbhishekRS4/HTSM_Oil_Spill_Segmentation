import torch
from torch import nn
from torch.nn import functional as F


class DeepLabV3(nn.Module):
    """
    DeepLabV3 class to build the DeepLabV3 decoder model

    ----------
    Attributes
    ----------
    in_channels : int
        number of input channels to decoder model from the encoder model's output
    num_classes : int
        number of classes for which the decoder needs to be built
    aspp_out_channels : int
        number of output channels of the ASPP layer (default: 256)
    final_out_channels : int
        number of output channels before applying classification conv layer (default: 256)
    aspp_dilate: list
        a list of dilation rates to be used for conv layers in ASPP block (default: [12, 24, 36])
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        aspp_out_channels=256,
        final_out_channels=256,
        aspp_dilate=[12, 24, 36],
    ):
        super().__init__()
        self.aspp_block = ASPPBlock(
            in_channels, aspp_dilate, aspp_out_channels=aspp_out_channels
        )
        self.classifier_conv_block = nn.Sequential(
            nn.Conv2d(aspp_out_channels, final_out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(final_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_out_channels, num_classes, 1, stride=1, padding="same"),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return

    def forward(self, encoded_features):
        """
        ---------
        Arguments
        ---------
        encoded_features : torch tensor
            a tensor of encoded features from the encoder

        -------
        Returns
        -------
        final_output_feature : torch tensor
            a tensor of final output logits
        """
        aspp_output_feature = self.aspp_block(encoded_features)
        final_output_feature = self.classifier_conv_block(aspp_output_feature)
        return final_output_feature


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3Plus class to build the DeepLabV3+ decoder model

    ----------
    Attributes
    ----------
    in_channels : int
        number of input channels to decoder model from the encoder model's output
    encoder_channels : int
        number of channels from the intermediate layer of the encoder for merging
    num_classes : int
        number of classes for which the decoder needs to be built
    encoder_projection_channels : int
        number of resulting projection channels from the intermediate layer of the encoder for merging (default: 48)
    aspp_out_channels : int
        number of output channels of the ASPP layer (default: 256)
    final_out_channels : int
        number of output channels before applying classification conv layer (default: 256)
    aspp_dilate: list
        a list of dilation rates to be used for conv layers in ASPP block (default: [12, 24, 36])
    """

    def __init__(
        self,
        in_channels,
        encoder_channels,
        num_classes,
        encoder_projection_channels=48,
        aspp_out_channels=256,
        final_out_channels=256,
        aspp_dilate=[12, 24, 36],
    ):

        super().__init__()
        self.projection_conv = nn.Sequential(
            nn.Conv2d(encoder_channels, encoder_projection_channels, 1, bias=False),
            nn.BatchNorm2d(encoder_projection_channels),
            nn.ReLU(inplace=True),
        )

        self.aspp_block = ASPPBlock(
            in_channels, aspp_dilate, aspp_out_channels=aspp_out_channels
        )

        self.classifier_conv_block = nn.Sequential(
            nn.Conv2d(
                aspp_out_channels + encoder_projection_channels,
                final_out_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(final_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_out_channels, num_classes, 1, stride=1, padding="same"),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return

    def forward(self, encoded_features, block_1_features):
        """
        ---------
        Arguments
        ---------
        encoded_features : torch tensor
            a tensor of encoded features from the encoder
        block_1_features : torch tensor
            a tensor of features from the intermediate layer from the encoder

        -------
        Returns
        -------
        final_output_feature : torch tensor
            a tensor of final output logits
        """
        encoder_connection = self.projection_conv(block_1_features)
        aspp_output_feature = self.aspp_block(encoded_features)
        aspp_output_feature = F.interpolate(
            aspp_output_feature,
            size=encoder_connection.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        final_output_feature = self.classifier_conv_block(
            torch.cat([encoder_connection, aspp_output_feature], dim=1)
        )
        return final_output_feature


class ASPPConvLayer(nn.Sequential):
    """
    ASPPConvLayer class to build the ASPPConvLayer used in ASPPBlock

    ----------
    Attributes
    ----------
    in_channels : int
        number of input channels to ASPPConvLayer
    out_channels : int
        number of output channels from ASPPConvLayer
    dilation : int
        dilation rate
    """

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return

    def forward(self, x):
        """
        ---------
        Arguments
        ---------
        x : torch tensor
            a tensor of input features

        -------
        Returns
        -------
        x : torch tensor
            output of the ASPPConvLayer
        """
        x = self.conv_block(x)
        return x


class ASPPPoolingLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        """
        ASPPPoolingLayer class to build the ASPPPoolingLayer used in ASPPBlock

        ----------
        Attributes
        ----------
        in_channels : int
            number of input channels to ASPPPoolingLayer
        out_channels : int
            number of output channels from ASPPPoolingLayer
        """
        super().__init__()
        self.avg_pool_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels, out_channels, 1, stride=1, padding="same", bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return

    def forward(self, x):
        """
        ---------
        Arguments
        ---------
        x : torch tensor
            a tensor of input features

        -------
        Returns
        -------
        x : torch tensor
            output of the ASPPPoolingLayer
        """
        size = x.shape[2:]
        x = self.avg_pool_block(x)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x


class ASPPBlock(nn.Module):
    def __init__(self, in_channels, atrous_rates, aspp_out_channels=256):
        """
        ASPPBlock class to build the ASPPBlock

        ---------
        Attributes
        ----------
        in_channels : int
            number of input channels to ASPPBlock
        atrous_rates : list
            list of dilation rates
        aspp_out_channels : int
            number of output channels of the ASPPBlock
        """
        super().__init__()

        self.aspp_init_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, aspp_out_channels, 1, stride=1, padding="same", bias=False
            ),
            nn.BatchNorm2d(aspp_out_channels),
            nn.ReLU(inplace=True),
        )

        modules = []
        modules.append(self.aspp_init_conv)
        modules += [
            ASPPConvLayer(in_channels, aspp_out_channels, atrous_rate)
            for atrous_rate in atrous_rates
        ]
        modules.append(ASPPPoolingLayer(in_channels, aspp_out_channels))
        self.aspp_module_layers = nn.ModuleList(modules)

        self.aspp_final_conv = nn.Sequential(
            nn.Conv2d(
                (2 + len(atrous_rates)) * aspp_out_channels,
                aspp_out_channels,
                1,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(aspp_out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return

    def forward(self, x):
        """
        ---------
        Arguments
        ---------
        x : torch tensor
            a tensor of input features

        -------
        Returns
        -------
        x : torch tensor
            output of the ASPPBlock
        """
        aspp_outputs = []
        for aspp_layer in self.aspp_module_layers:
            aspp_outputs.append(aspp_layer(x))
        concat_aspp_output = torch.cat(aspp_outputs, dim=1)
        final_aspp_output = self.aspp_final_conv(concat_aspp_output)
        return final_aspp_output
