import os
import sys
import time
import math
import argparse
import datetime
import numpy as np

import torch
from skimage.io import imsave
import torch.nn.functional as F

from metrics import *
from seg_models import *
from dataset import get_dataloader_for_inference


def create_directory(dir_path):
    """
    ---------
    Arguments
    ---------
    dir_path : str
        full directory path that needs to be created if it does not exist
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")
    return


def inference_loop(
    dataset_loader,
    list_images,
    model,
    dir_labels,
    dir_masks,
    num_classes,
    device,
    image_format=".png",
):
    """
    ---------
    Arguments
    ---------
    dataset_loader : object
        object of type dataloader
    list_images : list
        list of images for which the inference needs to be run
    model : object
        object of type model
    dir_labels : str
        full directory path to save prediction labels
    dir_masks : str
        full directory path to save prediction masks
    num_classes : int
        number of classes in the dataset
    device : str
        device on which inference needs to be run
    image_format : str
        the extension format of the images (default: ".png")
    """

    #  for lossless, always save labels and masks as png and not as jpeg
    model.eval()
    size = len(dataset_loader.dataset)
    num_batches = len(dataset_loader)
    infer_acc = 0
    infer_class_IOU = np.array([])

    dict_label_to_color_mapping = {
        0: np.array([0, 0, 0]),
        1: np.array([0, 255, 255]),
        2: np.array([255, 0, 0]),
        3: np.array([153, 76, 0]),
        4: np.array([0, 153, 0]),
    }

    cur_file_index = 0
    for image, label in dataset_loader:
        image = image.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)

        pred_logits = model(image)
        pred_probs = F.softmax(pred_logits, dim=1)
        pred_label = torch.argmax(pred_probs, dim=1)

        infer_acc += compute_mean_pixel_acc(label, pred_label)
        infer_class_IOU_cur_sample = compute_class_IOU(label, pred_label)

        if len(infer_class_IOU) == 0:
            infer_class_IOU = infer_class_IOU_cur_sample
        else:
            infer_class_IOU = np.vstack((infer_class_IOU, infer_class_IOU_cur_sample))

        pred_label_arr = pred_label.detach().cpu().numpy()
        pred_label_arr = np.squeeze(pred_label_arr)
        pred_label_one_hot = np.eye(num_classes)[pred_label_arr]

        pred_mask_arr = np.zeros((pred_label_arr.shape[0], pred_label_arr.shape[1], 3))
        for sem_class in range(num_classes):
            curr_class_label = pred_label_one_hot[:, :, sem_class]
            curr_class_label = curr_class_label.reshape(
                pred_label_one_hot.shape[0], pred_label_one_hot.shape[1], 1
            )

            curr_class_color_mapping = dict_label_to_color_mapping[sem_class]
            curr_class_color_mapping = curr_class_color_mapping.reshape(
                1, curr_class_color_mapping.shape[0]
            )

            pred_mask_arr += curr_class_label * curr_class_color_mapping

        pred_label_arr = pred_label_arr.astype(np.uint8)
        pred_mask_arr = pred_mask_arr.astype(np.uint8)

        file_pred_label = os.path.join(
            dir_labels, list_images[cur_file_index].replace(".jpg", image_format)
        )
        file_pred_mask = os.path.join(
            dir_masks, list_images[cur_file_index].replace(".jpg", image_format)
        )

        padded_height, padded_width = pred_label_arr.shape

        # remove padding and save the label and mask images
        imsave(
            file_pred_label,
            pred_label_arr[11 : padded_height - 11, 15 : padded_width - 15],
        )
        imsave(
            file_pred_mask,
            pred_mask_arr[11 : padded_height - 11, 15 : padded_width - 15],
        )

        cur_file_index += 1

    infer_acc /= num_batches
    infer_per_class_IOU = np.nanmean(infer_class_IOU, axis=0)
    return infer_acc, infer_per_class_IOU


def run_inference(FLAGS):
    inference_dataset_loader, list_inference_images = get_dataloader_for_inference(
        FLAGS.dir_dataset
    )
    print("dataset information")
    print(f"number of test samples: {len(list_inference_images)}")

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    oil_spill_seg_model.load_state_dict(torch.load(FLAGS.file_model_weights))

    dir_labels = os.path.join(FLAGS.dir_save_preds, "labels")
    dir_masks = os.path.join(FLAGS.dir_save_preds, "masks")

    create_directory(dir_labels)
    create_directory(dir_masks)

    infer_acc, infer_per_class_IOU = inference_loop(
        inference_dataset_loader,
        list_inference_images,
        oil_spill_seg_model,
        dir_labels,
        dir_masks,
        FLAGS.num_classes,
        device,
    )
    infer_acc = infer_acc * 100
    infer_per_class_IOU = infer_per_class_IOU * 100
    infer_IOU = np.mean(infer_per_class_IOU)
    print("Inference test set metrics")
    print(f"accuracy: {infer_acc:.3f} %")
    print(f"mean IOU: {infer_IOU:.3f} %")
    print(f"per class IOU")
    print(infer_per_class_IOU)
    return


def main():
    dir_dataset = (
        "/home/abhishek/Desktop/RUG/htsm_masterwork/oil-spill-detection-dataset/"
    )
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
    file_model_weights = "/home/abhishek/Desktop/RUG/htsm_masterwork/resnet_patch_padding_sgd/fold_1/resnet_18_deeplab_v3+/oil_spill_seg_resnet_18_deeplab_v3+_98.pt"
    dir_save_preds = "./fold_1_resnet_18_deeplab_v3+_98/"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--pretrained",
        default=1,
        type=int,
        choices=[0, 1],
        help="use pretrained encoder (1:True, 0:False)",
    )
    parser.add_argument(
        "--dir_dataset",
        default=dir_dataset,
        type=str,
        help="full directory path to the dataset",
    )
    parser.add_argument(
        "--num_classes",
        default=num_classes,
        type=int,
        help="number of semantic classes in the dataset",
    )
    parser.add_argument(
        "--which_model",
        default=which_model,
        type=str,
        choices=list_model_choices,
        help="which model to train",
    )
    parser.add_argument(
        "--file_model_weights",
        default=file_model_weights,
        type=str,
        help="full path to the model weights file ",
    )
    parser.add_argument(
        "--dir_save_preds",
        default=dir_save_preds,
        type=str,
        help="full directory path to save the predictions",
    )

    FLAGS, unparsed = parser.parse_known_args()
    run_inference(FLAGS)
    return


if __name__ == "__main__":
    main()
