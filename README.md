# HTSM Masterwork - Oil Spill Detection using Semantic Segmentation


## Required dependencies
* To install pytorch use the following command
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
* The other required dependencies are available in [requirements.txt](requirements.txt)

## Info
* This repo contains the project work done as part of the [HTSM Masterwork](https://www.rug.nl/education/honours-college/htsm-masterprogramme/about-the-programme) at [University of Groningen](https://www.rug.nl/)

## Dataset information
* The dataset used in this Masterwork project can be found [here](https://m4d.iti.gr/oil-spill-detection-dataset/)

## Instructions to run the code
* For any python script, use the following to list all the command-line options
```
python3 script_name.py --help
```
* Run the script [src/compute_stats_for_train.py](src/compute_stats_for_train.py) to
compute the statistics for training set. This will generate [src/image_stats.json](src/image_stats.json)
with the mean and std. dev. of images in the training set.
* To train the models, run the script [src/train.py](src/train.py)
* To compute the k-fold validation metrics, run the script [src/compute_kfold_validation_metrics.py](src/compute_kfold_validation_metrics.py)
* Run the notebook [src/EDA.ipynb](src/EDA.ipynb) or the script [src/exploratory_data_analysis.py](src/exploratory_data_analysis.py) to generate the plot of class distribution in the dataset.
* For plotting the variation of loss, accuracy, IoU during the training phase, use the notebook [src/plot_graphs.ipynb](src/plot_graphs.ipynb)
* To infer on the test set, run the script [src/inference.py](src/inference.py)

## Sample predictions
![Sample predicted mask 1](images/pred_mask_img_0001.png?raw=true)
![Sample predicted mask 2](images/pred_mask_img_0007.png?raw=true)
![Sample predicted mask 3](images/pred_mask_img_0035.png?raw=true)
![Sample predicted mask 4](images/pred_mask_img_0054.png?raw=true)
![Sample predicted mask 5](images/pred_mask_img_0105.png?raw=true)

## References
* [HTSM Masterwork](https://www.rug.nl/education/honours-college/htsm-masterprogramme/about-the-programme)
* [https://m4d.iti.gr/oil-spill-detection-dataset/](https://m4d.iti.gr/oil-spill-detection-dataset/)
