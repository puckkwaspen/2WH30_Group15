# 📝 Image Classification Project
- **Written by**: [Puck Kwaspen, Alexandra Liskayova, Jagoda Nawrat, Cosmin Cosniceanu, Kristiyan Valev, Klaas Biekens](#)
---

## 📚 Overview

This project explores the use of convolutional neural networks (CNNs) in image classification task.
It deals with binary classification of images into the following categories: plastic and non-plastic.
This was based on a dataset of 1000 photographed images on a uniform black background with their names,
surface properties and other categories being recorded. Furthermore, it explores the CNN code
structure with other datasets to establish the accuracy of this model with respect to the small dataset.

### 📄 📂 Data Files and Folders
- `annotations_final.csv`: Original annotations provided for the task. The material category is recorded in this file.
- `annotations_other.csv`: Annotation file for experiment number 2.
- `i190_data`: A part of the original dataset provided with two chosen lighting versions.
- `images_other`: A part of a TrashNet dataset.

### 📈 Model Results
- `results_log.csv`: Contains the results from the 5-fold cross validation experiments. Experimental results that are of
the main CNN model start with a number in the results, the results from the other experiments start with a "
- `test_dataset_result.csv`: Contains the results of the best hyperparameters found in each iteration of the experiment evaluated
on the test set. Experimental results that are of the main CNN model start with 'Experiment 1',
 the results from the other experiments start with 'Experiment 100'

Note: not all experiment iterations are present in the `results_log.csv` file as the file was implemented only after some of the
experiments were conducted and not on every computer where the experiments were run.
---

## 🚀 Getting Started

### 🔧 Requirements
1. **Python Version**: `3.10+`
2. **Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Hardware**:
   - **CPU**: Multi-core processor
   - **GPU (Optional)**: NVIDIA CUDA-supported for accelerated training

4. **Dataset**:
   Ensure that the `annotations_final.csv` is found in the directory
   Ensure that `i190_data` folder is found in the directory- this file contains both the i190 and i130 lighting images
   as prepared in the file `Extraction Of Images.py` - it is not necessary to run this file if the folder `i190_data` is visible.
   If not, make sure to download the image folder `aloi_red4_col` and run the `Extraction Of Images.py` to create the
   `i190_data`. If you run the `Extraction Of Images.py` file, make sure to change the paths according to your setup.

---

## 💡 How to Run

### 🛠️ The main experiment

1. Download the requirements: pip install -r requirements.txt as mentioned above.
2. Run the `data_preparation.py` file. For this, make sure to have `annotations_final.csv` downloaded in the directory.
3. Run the `CNN.py` python script. For reproducibility, the seed is set to 678. You can change the number of iterations in
the code - line XXX to reduce the running time. Two iterations take approximately 15 minutes to run. However, the more iterations
you choose the run, the greater the chance of finding the most effective combination of hyperparameters.
4. The results of each iteration are stored in `results_log.csv` so you can see the evaluation metrics of each iteration there.
Moreover, from each iteration the best combination of hyperparameters is used to retrain the model and evaluate this setting on the
test set. These results are stored in `test_dataset_result.csv`.


### 🚀 Other experiment:
1. Make sure the `images_other` folder is in the directory. This dataset is called TrashNet and was downloaded from:
https://www.kaggle.com/datasets/feyzazkefe/trashnet
This dataset was manually modified so that the number of plastic and non-plastic objects is the same as well as a
variety of the other classes is approximately the same. Therefore, we advise to use the dataset given here and not to
use the original one.
2. Run the `data_preparation2.py` file. For this, make sure to have `annotations_other.csv` downloaded in the directory.
3. Run the `CNN2.py` python script. For reproducibility, the seed is set to 678. You can change the number of iterations in
the code - line XXX to reduce the running time. Two iterations take approximately 15 minutes to run. However, the more iterations
you choose the run, the greater the chance of finding the most effective combination of hyperparameters.
4. The results of each iteration are stored in `results_log.csv` so you can see the evaluation metrics of each iteration there.
Moreover, from each iteration the best combination of hyperparameters is used to retrain the model and evaluate this setting on the
test set. These results are stored in `test_dataset_result.csv`. To know how to differentiate between the results,
see section 'Model Results' of this README.

### 🚀 Other files:

Additionaly, you can find other files in this respiratory. It is not necessary to run these for the functioning of the model
and the execution of the task. These were used by the team to better understand the data and the task at hand.
These files are all placed in the folder 'Additional'.

The folder 'data' contains the images creates synthetically with SMOTE.