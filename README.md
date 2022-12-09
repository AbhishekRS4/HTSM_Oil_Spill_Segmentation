# HTSM Masterwork - Oil Spill Detection using Semantic Segmentation

# Info
* This repo contains the project work done as part of the [HTSM Masterwork](https://www.rug.nl/education/honours-college/htsm-masterprogramme/about-the-programme) at [University of Groningen](https://www.rug.nl/)

## Instructions
* Use the following to list all the command-line options
```
python3 script_name.py --help
```
* Run the script [src/compute_stats_for_train.py](src/compute_stats_for_train.py) to
compute the statistics for training set. This will generate [src/image_stats.json](src/image_stats.json)
with the mean and std. dev. of images in the training set.
* To train the models, run the script [src/train.py](src/train.py)
