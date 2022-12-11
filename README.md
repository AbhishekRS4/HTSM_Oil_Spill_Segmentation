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
* The dataset used in this Master project can be found [here](https://m4d.iti.gr/oil-spill-detection-dataset/)

## Instructions
* Use the following to list all the command-line options
```
python3 script_name.py --help
```
* Run the script [src/compute_stats_for_train.py](src/compute_stats_for_train.py) to
compute the statistics for training set. This will generate [src/image_stats.json](src/image_stats.json)
with the mean and std. dev. of images in the training set.
* To train the models, run the script [src/train.py](src/train.py)
