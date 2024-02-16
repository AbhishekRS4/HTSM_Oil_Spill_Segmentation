# Oil Spill Segmentation using Deep Encoder-Decoder models
##### [Abhishek Ramanathapura Satyanarayana](https://orcid.org/0009-0003-1248-0988), [Maruf A. Dhali](https://orcid.org/0000-0002-7548-3858)


## Paper available on ArXiv
* The results of this research is available in the [ArXiv paper](https://arxiv.org/abs/2305.01386).


## Info about the project
* This repo contains the project work done as part of the [HTSM Masterwork](https://www.rug.nl/education/honours-college/htsm-masterprogramme/about-the-programme) at [University of Groningen](https://www.rug.nl/).
* Research work carried as part of the HTSM Masterwork to train deep learning CNN model for segmentation task to detect oil spills from the satellite
Synthetic Aperture Radar (SAR) data.
* This research is towards application of AI for good, particularly towards conservation of nature with AI.
* This project was done under the supervision of [Mr. Maruf A. Dhali](https://www.rug.nl/staff/m.a.dhali/)


## Dataset information
* The details about the dataset used in this Masterwork project can be found here - [dataset details](https://m4d.iti.gr/oil-spill-detection-dataset/).
* This dataset contains labels for 5 classes --- sea surface, oil spill, oil spill look-alike, ship, and land.
* ThIs dataset is a relatively smaller dataset when compared to other popular benchmark datasets for the segmentation task.


## Required dependencies for training
* To install pytorch use the following command
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
* The other required dependencies for training are available in [requirements.txt](requirements.txt).


## Instructions to run the code for training
* For any python script, use the following to list all the command-line options
```
python3 script_name.py --help
```
* Run the script [src/training/compute_stats_for_train.py](src/training/compute_stats_for_train.py) to
compute the statistics for training set. This will generate [src/training/image_stats.json](src/training/image_stats.json)
with the mean and std. dev. of images in the training set.
* To train the models, run the script [src/training/train.py](src/training/train.py).
* To compute the k-fold validation metrics, run the script [src/training/compute_kfold_validation_metrics.py](src/training/compute_kfold_validation_metrics.py).
* Run the notebook [src/training/EDA.ipynb](src/training/EDA.ipynb) or the script [src/training/exploratory_data_analysis.py](src/training/exploratory_data_analysis.py) to generate the plot of class distribution in the dataset.
* For plotting the variation of loss, accuracy, IoU during the training phase, use the notebook [src/training/plot_graphs.ipynb](src/training/plot_graphs.ipynb).
* To infer on the test set, run the script [src/training/inference.py](src/training/inference.py).


## Docker deployment instructions
* The streamlit [app](src/app.py) has been developed for deployment.
* The detailed python package requirements for the streamlit app can be found in [src/requirements_deployment.txt](src/requirements_deployment.txt).
* To build the container, run the following command inside `src` directory
```
docker build -t app_oil_spill .
```
* To the run the container, run the following command inside `src` directory
```
docker run -p 8000:8000 -t app_oil_spill
```


## Huggingface deployment
* A streamlit application, with the best performing model, has been deployed to [Huggingface](https://huggingface.co/spaces/abhishekrs4/Oil_Spill_Segmentation)


## Qualitative results - sample test set predictions
![Sample predicted mask 1](images/pred_mask_img_0001.png?raw=true)
![Sample predicted mask 2](images/pred_mask_img_0007.png?raw=true)
![Sample predicted mask 3](images/pred_mask_img_0035.png?raw=true)
![Sample predicted mask 4](images/pred_mask_img_0054.png?raw=true)
![Sample predicted mask 5](images/pred_mask_img_0105.png?raw=true)
* Since the dataset is not publicly available, the original test set images from the dataset are not uploaded but only their predictions are included in the repo.


## Quantitative results
* The best model's class-wise and mean IoU performance is presented below.

class  |  class IoU (%)  |
-------|-----------------|
sea surface  |  96.422  |
**oil spill**  |  **61.549**  |
oil spill look-alike  |  40.773  |
ship  |  33.378  |
land  |  92.218  |
**mean IoU**  |  **64.868**  |


## Sphinx docstring generation
* The following are the steps to generate docstrings using sphinx
* Create a directory named `docs` and go to that directory
```
mkdir docs && cd docs
```
* Run the everyone of following commands inside `docs` directory
* Run the following command with appropriate options
```
sphinx-quickstart ..
```
* Open the file `index.rst` and add `modules` to it
* In the file `conf.py`, make the following changes
  * set `html_theme = 'sphinx_rtd_theme'`
  * set `extensions = ["sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc"]`
  * add the following to the beginning of the `conf.py` file
  ```
  import os
  import sys

  sys.path.insert(0, os.path.abspath(".."))
  ```
* Run the following command
```
sphinx-apidoc -o . ..
```
* Create html files with documentation
```
make html
```


## References
* [HTSM Masterwork](https://www.rug.nl/education/honours-college/htsm-masterprogramme/about-the-programme)
* [https://m4d.iti.gr/oil-spill-detection-dataset/](https://m4d.iti.gr/oil-spill-detection-dataset/)
