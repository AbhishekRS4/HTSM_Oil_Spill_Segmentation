import torch
from torch import nn
from torch.nn import functional as F

class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels, encoder_channels, num_classes, encoder_projection_channels=48,
        aspp_out_channels=256, final_out_channels=256, aspp_dilate=[12, 24, 36]):

        super().__init__()
        self.projection_conv = nn.Sequential(
            nn.Conv2d(encoder_channels, encoder_projection_channels, 1, bias=False),
            nn.BatchNorm2d(encoder_projection_channels),
            nn.ReLU(inplace=True),
        )

        self.aspp_block = ASPPBlock(in_channels, aspp_dilate, aspp_out_channels=aspp_out_channels)

        self.classifier_conv_block = nn.Sequential(
            nn.Conv2d(
                aspp_out_channels + encoder_projection_channels,
                final_out_channels, 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(final_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_out_channels, num_classes, 1, stride=1, padding="same")
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
        encoder_connection = self.projection_conv(block_1_features)
        aspp_output_feature = self.aspp_block(encoded_features)
        aspp_output_feature = F.interpolate(
            aspp_output_feature, size=encoder_connection.shape[2:],
            mode="bilinear", align_corners=False
        )
        final_output_feature = self.classifier_conv_block(
            torch.cat([encoder_connection, aspp_output_feature], dim=1)
        )
        return final_output_feature

class ASPPConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, padding=dilation,
                dilation=dilation, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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
        x = self.conv_block(x)
        return x

class ASPPPoolingLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avg_pool_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding="same", bias=False),
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
        size = x.shape[2:]
        x = self.avg_pool_block(x)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x

class ASPPBlock(nn.Module):
    def __init__(self, in_channels, atrous_rates, aspp_out_channels=256):
        super().__init__()

        self.aspp_init_conv = nn.Sequential(
            nn.Conv2d(in_channels, aspp_out_channels, 1, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(aspp_out_channels),
            nn.ReLU(inplace=True)
        )

        modules = []
        modules.append(self.aspp_init_conv)
        modules += [
            ASPPConvLayer(in_channels, aspp_out_channels, atrous_rate) for atrous_rate in atrous_rates
        ]
        modules.append(ASPPPoolingLayer(in_channels, aspp_out_channels))
        self.aspp_module_layers = nn.ModuleList(modules)

        self.aspp_final_conv = nn.Sequential(
            nn.Conv2d((2 + len(atrous_rates)) * aspp_out_channels,
                aspp_out_channels, 1, stride=1, padding="same", bias=False
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
        aspp_outputs = []
        for aspp_layer in self.aspp_module_layers:
            aspp_outputs.append(aspp_layer(x))
        concat_aspp_output = torch.cat(aspp_outputs, dim=1)
        final_aspp_output = self.aspp_final_conv(concat_aspp_output)
        return final_aspp_output
