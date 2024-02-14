import os
import argparse
import numpy as np
import pandas as pd


def compute_kfold_validation_metrics(FLAGS):
    print(f"attempting to read results data from the directory: {FLAGS.dir_results}")
    print(f"for model: {FLAGS.which_model}")
    dir_k_folds = os.listdir(FLAGS.dir_results)
    max_validation_IOUs = np.array([])
    max_validation_accs = np.array([])

    for dir_fold in dir_k_folds:
        cur_dir_fold = os.path.join(FLAGS.dir_results, dir_fold, FLAGS.which_model)

        if os.path.isdir(cur_dir_fold):
            df_metrics = pd.read_csv(os.path.join(cur_dir_fold, FLAGS.file_metrics))
            max_index = np.argmax(df_metrics["valid_IOU"].to_numpy())
            max_validation_IOUs = np.append(
                max_validation_IOUs, df_metrics["valid_IOU"].to_numpy()[max_index]
            )
            max_validation_accs = np.append(
                max_validation_accs, df_metrics["valid_acc"].to_numpy()[max_index]
            )

    max_validation_IOUs = 100 * max_validation_IOUs
    max_validation_accs = 100 * max_validation_accs
    num_folds = len(max_validation_IOUs)
    print(f"Number of folds = {num_folds}")
    print("Validation mIOUs")
    print(max_validation_IOUs)
    print("Validation pixelwise accuracy")
    print(max_validation_accs)
    IOU_mean = np.mean(max_validation_IOUs)
    IOU_std = np.std(max_validation_IOUs)
    acc_mean = np.mean(max_validation_accs)
    acc_std = np.std(max_validation_accs)
    print(f"validation IOU: {IOU_mean:.3f} +/- {IOU_std:.3f} %")
    print(f"validation acc: {acc_mean:.3f} +/- {acc_std:.3f} %")
    return


def main():
    dir_results = "/home/abhishek/Desktop/RUG/htsm_masterwork/resnet_patch_padding_sgd/"
    file_metrics = "train_metrics.csv"
    which_model = "resnet_18_deeplab_v3+"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dir_results",
        default=dir_results,
        type=str,
        help="full directory path to the results",
    )
    parser.add_argument(
        "--file_metrics",
        default=file_metrics,
        type=str,
        help="csv file name with train/validation metrics",
    )
    parser.add_argument(
        "--which_model",
        default=which_model,
        type=str,
        help="model for which the kfold validation metrics needs to be computed",
    )

    FLAGS, unparsed = parser.parse_known_args()
    compute_kfold_validation_metrics(FLAGS)
    return


if __name__ == "__main__":
    main()
