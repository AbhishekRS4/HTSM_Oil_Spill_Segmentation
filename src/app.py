import os
import numpy as np
import streamlit as st

import torch
from PIL import Image
from skimage.io import imread
import torch.nn.functional as F

from training.metrics import *
from training.seg_models import *
from training.image_preprocessing import ImagePadder
from training.logger_utils import load_dict_from_json
from training.dataset import get_dataloader_for_inference

def run_inference(image_array, file_weights, num_classes=5, file_stats_json="training/image_stats.json"):
    oil_spill_seg_model = ResNet50DeepLabV3Plus(
        num_classes=num_classes, pretrained=True
    )
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oil_spill_seg_model.to(device)
    oil_spill_seg_model.load_state_dict(torch.load(file_weights, map_location=device))
    oil_spill_seg_model.eval()

    dict_label_to_color_mapping = {
        0: np.array([0, 0, 0]),
        1: np.array([0, 255, 255]),
        2: np.array([255, 0, 0]),
        3: np.array([153, 76, 0]),
        4: np.array([0, 153, 0]),
    }

    try:
        dict_stats = load_dict_from_json(file_stats_json)
    except:
        dir_json = os.path.dirname(os.path.realpath(__file__))
        dict_stats = load_dict_from_json(os.path.join(dir_json, file_stats_json))

    try:
        image_padder = ImagePadder("/data/images")
    except:
        image_padder = ImagePadder("./sample_padding_image_for_inference")

    # apply padding and preprocessing
    image_padded = image_padder.pad_image(image_array)
    image_preprocessed = image_padded / 255.
    image_preprocessed = image_preprocessed - dict_stats["mean"]
    image_preprocessed = image_preprocessed / dict_stats["std"]
    image_preprocessed = np.expand_dims(image_preprocessed, axis=0)
    # NCHW format
    image_preprocessed = np.transpose(image_preprocessed, (0, 3, 1, 2))


    image_tensor = torch.tensor(image_preprocessed).float()
    image_tensor = image_tensor.to(device, dtype=torch.float)
    pred_logits = oil_spill_seg_model(image_tensor)
    pred_probs = F.softmax(pred_logits, dim=1)
    pred_label = torch.argmax(pred_probs, dim=1)

    pred_label_arr = pred_label.detach().cpu().numpy()
    pred_label_arr = np.squeeze(pred_label_arr)
    pred_label_one_hot = np.eye(num_classes)[pred_label_arr]

    pred_mask_arr = np.zeros((pred_label_arr.shape[0], pred_label_arr.shape[1], 3))
    for sem_class in range(num_classes):
        curr_class_label = pred_label_one_hot[:, :, sem_class]
        curr_class_label = curr_class_label.reshape(pred_label_one_hot.shape[0], pred_label_one_hot.shape[1], 1)

        curr_class_color_mapping = dict_label_to_color_mapping[sem_class]
        curr_class_color_mapping = curr_class_color_mapping.reshape(1, curr_class_color_mapping.shape[0])

        pred_mask_arr +=  curr_class_label * curr_class_color_mapping

    pred_label_arr = pred_label_arr.astype(np.uint8)
    pred_mask_arr = pred_mask_arr.astype(np.uint8)
    padded_height, padded_width = pred_label_arr.shape
    pred_mask_arr = pred_mask_arr[11:padded_height-11, 15:padded_width-15]

    return pred_mask_arr

def infer():
    st.title("Oil spill detection app")

    # file_weights_default = "/home/abhishek/Desktop/RUG/htsm_masterwork/resnet_patch_padding_sgd/fold_5/resnet_50_deeplab_v3+/oil_spill_seg_resnet_50_deeplab_v3+_80.pt"
    file_weights_default = "/data/models/oil_spill_seg_resnet_50_deeplab_v3+_80.pt"
    file_weights = st.sidebar.text_input("File model weights", file_weights_default)

    if not os.path.isfile(file_weights):
        st.write("Wrong weights file path")
    else:
        st.write(f"Weights file: {file_weights}")

    # select an input SAR image file
    image_file_buffer = st.sidebar.file_uploader("Select input SAR image", type=["jpg", "jpeg"])
    # read the image
    if image_file_buffer is not None:
        image = Image.open(image_file_buffer)
        image_array = np.array(image)
        st.image(image_array, caption=f"Input image: {image_file_buffer.name}")
    else:
        st.write("Input image: not selected")

    # select a mask image file
    mask_file_buffer = st.sidebar.file_uploader("Select mask image", type=["png"])
    # read the mask
    if mask_file_buffer is not None:
        mask = Image.open(mask_file_buffer)
        mask_array = np.array(mask)
        st.image(mask_array, caption=f"Mask image: {mask_file_buffer.name}")
    else:
        st.write("Mask image: not selected")

    # run inference when the option is invoked by the user
    infer_button = st.sidebar.button("Run inference")
    if infer_button:
        mask_predicted = run_inference(image_array, file_weights)
        st.image(mask_predicted, caption=f"Predicted mask for the input: {image_file_buffer.name}")
    return

def main():
    infer()
    return

if __name__ == "__main__":
    main()
