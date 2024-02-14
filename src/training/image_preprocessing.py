import os
import cv2
import numpy as np
from skimage.io import imread


class ImagePadder:
    def __init__(
        self,
        dir_images,
        pad_left=15,
        pad_right=15,
        pad_top=11,
        pad_bottom=11,
        file_anchor_image="img_0814.jpg",
    ):
        """
        ImagePadder class for padding images

        ----------
        Attributes
        ----------
        dir_images : str
            full directory path containing images
        pad_left : int
            number of pixels to be padded to the left of the input image (default: 15)
        pad_right : int
            number of pixels to be padded to the right of the input image (default: 15)
        pad_top : int
            number of pixels to be padded to the top of the input image (default: 11)
        pad_bottom : int
            number of pixels to be padded to the bottom of the input image (default: 11)
        file_anchor_image : str
            file with anchor image whose pixels will be used as a reference for padding (default: "img_0814.jpg")
        """
        self._anchor_image = imread(os.path.join(dir_images, file_anchor_image))
        self._anchor_image_shape = self._anchor_image.shape
        self._pad_left = pad_left
        self._pad_right = pad_right
        self._pad_top = pad_top
        self._pad_bottom = pad_bottom
        self._anchor_image_resized = None
        self._anchor_image_resized_shape = None

        self._set_anchor_image_resized()

    def _set_anchor_image_resized(self):
        anchor_image_shape = self._anchor_image.shape
        height, width = anchor_image_shape[0], anchor_image_shape[1]
        target_width = self._pad_left + width + self._pad_right
        target_height = self._pad_top + height + self._pad_bottom
        # print(target_width, target_height)
        self._anchor_image_resized = cv2.resize(
            self._anchor_image[:, 260:, :],
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
        self._anchor_image_resized_shape = self._anchor_image_resized.shape
        # print(self._anchor_image_resized_shape)
        return

    def pad_image(self, image):
        padded_image = self._anchor_image_resized
        padded_image[
            self._pad_top : self._anchor_image_resized_shape[0] - self._pad_bottom,
            self._pad_left : self._anchor_image_resized_shape[1] - self._pad_right,
            :,
        ] = image
        return padded_image

    def pad_label(self, label):
        padded_label = np.pad(
            label,
            ((self._pad_top, self._pad_bottom), (self._pad_left, self._pad_right)),
        )
        return padded_label
