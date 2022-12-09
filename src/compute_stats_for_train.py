import os
import json
import argparse
import numpy as np
from skimage.io import imread

def write_json_file(file_json, dict_data):
    """
    ---------
    Arguments
    ---------
    file_json : str
        full path of json file to be saved
    dict_data : dict
        dictionary of params to be saved in the json file
    """
    with open(file_json, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(dict_data, indent=4))
    return

def compute_stats(FLAGS):
    list_images = sorted(os.listdir(FLAGS.dir_images))

    num_images = len(list_images)
    print(f"Num images: {num_images}")
    print("Computing statistics for all the training images in the dataset")

    all_means = np.array([])
    all_stds = np.array([])

    for idx in range(num_images):
        image = imread(os.path.join(FLAGS.dir_images, list_images[idx]))
        image = image / 255.0
        """
        if idx == 10:
            break
        """
        all_means = np.append(all_means, np.mean(image[:, :, 0]))
        all_stds = np.append(all_stds, np.std(image[:, :, 0]))

    mean_of_images = np.mean(all_means)
    std_of_images = np.sqrt(all_stds.shape[0] * np.sum(np.square(all_stds)) / (all_stds.shape[0] - 1)**2)
    print(f"mean: {mean_of_images:.4f}, std: {std_of_images:.4f}")
    dict_stats = {}
    dict_stats["mean"] = round(mean_of_images, 4)
    dict_stats["std"] = round(std_of_images, 4)
    write_json_file(FLAGS.file_json, dict_stats)
    print(f"Training image statistics saved in {FLAGS.file_json}")
    print("Completed computing statistics for all the training images in the dataset")
    return

def main():
    file_json = "image_stats.json"
    dir_images = "/home/abhishek/Desktop/RUG/htsm_masterwork/oil-spill-detection-dataset/train/images/"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--file_json", default=file_json,
        type=str, help="full path of json file to be saved")
    parser.add_argument("--dir_images", default=dir_images,
        type=str, help="full path of directory containing training images")

    FLAGS, unparsed = parser.parse_known_args()
    compute_stats(FLAGS)
    return

if __name__ == "__main__":
    main()
