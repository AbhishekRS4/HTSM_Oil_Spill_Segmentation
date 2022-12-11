import sys
import torch
import argparse
from torchsummary import summary

from seg_models import *


def summarize_model(FLAGS):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA device not found, so exiting....")
        sys.exit(0)

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

    print(f"oil spill segmentation model name - {FLAGS.which_model}")
    print(f"image input - {FLAGS.input_channels} x {FLAGS.image_height} x {FLAGS.image_width}")
    print("\n\nmodel summary\n")
    oil_spill_seg_model = oil_spill_seg_model.to(device)
    print(summary(oil_spill_seg_model, (FLAGS.input_channels, FLAGS.image_height, FLAGS.image_width)))
    return

def main():
    image_height = 672
    image_width = 1280
    input_channels = 3
    pretrained = 1
    num_classes = 5
    which_model = "resnet_18_deeplab_v3+"
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

    parser.add_argument("--which_model", default=which_model,
        type=str, choices=list_model_choices, help="the model for which summary needs to be generated")
    parser.add_argument("--image_height", default=image_height,
        type=int, help="image height")
    parser.add_argument("--image_width", default=image_width,
        type=int, help="image width")
    parser.add_argument("--num_classes", default=num_classes,
        type=int, help="number of classes")
    parser.add_argument("--pretrained", default=pretrained,
        type=int, choices=[1, 0], help="pretrained [1 - True, 0 - False]")
    parser.add_argument("--input_channels", default=input_channels,
        type=int, choices=[1, 3, 4], help="number of input channels [1 - Depth, 3 - RGB, 4 - RGBD]")

    FLAGS, unparsed = parser.parse_known_args()
    summarize_model(FLAGS)
    return

if __name__ == "__main__":
    main()
