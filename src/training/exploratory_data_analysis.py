import os
import argparse
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt


def do_exploratory_data_analysis(FLAGS):
    dir_labels = os.path.normpath(FLAGS.dir_labels)
    print(f"Label files are read from the directory: {dir_labels}")

    list_label_files = os.listdir(dir_labels)
    num_label_files = len(list_label_files)

    print(f"Number of label files : {num_label_files}")
    dict_label_mapping = {
        0: "Sea Surface",
        1: "Oil Spill",
        2: "Look-alike",
        3: "Ship",
        4: "Land",
    }

    dict_class_counts = {}
    for file_label in list_label_files:
        label = imread(os.path.join(dir_labels, file_label))
        unique, unique_counts = np.unique(label, return_counts=True)

        for key, value in zip(unique, unique_counts):
            if key in dict_class_counts.keys():
                dict_class_counts[key] += value
            else:
                dict_class_counts[key] = value

    print(dict_class_counts)
    expr = r"$\times 10^3$"
    fig = plt.figure(figsize=(12, 12))
    plt.bar(
        list(dict_label_mapping.values()),
        np.array(list(dict_class_counts.values())) / 1000,
    )
    plt.grid()
    plt.title("Class distribution for Oil Spill Detection Dataset", fontsize=16)
    plt.xlabel("Semantic class labels", fontsize=20)
    plt.ylabel(f"Label counts ({expr})", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    return


def main():
    dir_labels = "/home/abhishek/Desktop/RUG/htsm_masterwork/oil-spill-detection-dataset/train/labels_1D/"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dir_labels",
        default=dir_labels,
        type=str,
        help="full directory path to the labels",
    )
    FLAGS, unparsed = parser.parse_known_args()
    do_exploratory_data_analysis(FLAGS)
    return


if __name__ == "__main__":
    main()
