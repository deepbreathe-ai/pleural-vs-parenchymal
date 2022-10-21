# Pleural vs Parenchymal Lung Ultrasound Classifier
![Deep Breathe Logo](img/readme/deep-breathe-logo.jpg "Deep Breath AI")   

We at [Deep Breathe](https://www.deepbreathe.ai/) sought to train a deep learning model for the task
of automating the distinction between parenchymal and pleural lung ultrasound videos.

This repository contains work relating to development and validation of a pleural vs parenchymal
ultrasound view image classifier that was used for the creation of the paper
[Enhancing Annotation Efficiency with Machine Learning: Automated Partitioning of a Lung Ultrasound Dataset by View](https://www.mdpi.com/1856464).

[comment]: <> (TODO: Update table of contents to use corrent links and section titles.)
## Table of Contents
1. [**_Getting Started_**](#getting-started)
2. [**_Building a Dataset_**](#building-a-dataset)
3. [**_Use Cases_**](#use-cases)  
   i)[**_Train Single Experiment_**](#train-single-experiment)  
   ii) [**_K-Fold Cross Validation_**](#k-fold-cross-validation)  
   iii) [**_Hyper Parameter Optimization_**](#hyper-parameter-optimization)  
   iv) [**_Predictions_**](#predictions)  
   v) [**_Grad-CAM for Individual Frame Predictions_**](#grad-cam-for-individual-frame-predictions)  
4. [**_Project Configuration_**](#project-configuration)
5. [**_Project Structure_**](#project-structure)
6. [**_Contacts_**](#contacts)

[comment]: <> (TODO: Update the getting started section to refplect the project's specific setup.)
## Getting Started
1. Clone this repository (for help see this
   [tutorial](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)).
2. Install the necessary dependencies (listed in
   [requirements.txt](requirements.txt)). To do this, open a terminal in
   the root directory of the project and run the following:
   ```
   $ pip install -r requirements.txt
   ```
3. Obtain lung ultrasound data and preprocess it accordingly. See
   [building a dataset](#building-a-dataset) for more details.
   
4. Update the _TRAIN >> MODEL_DEF_ field of [_config.yml_](config.yml) with
   the appropriate string representing the model type you wish to
   train. To train a model, ensure the _TRAIN >>
   EXPERIMENT_TYPE_ field is set to _'single_train'_.
5. Execute [_train.py_](src/train.py) to train your chosen model on your
   preprocessed data. The trained model will be serialized within
   _results/models/_, and its filename will resemble the following
   structure: _model{yyyymmdd-hhmmss}.h5_, where _{yyyymmdd-hhmmss}_ is the current
   time.
6. Navigate to _results/logs/_ to see the tensorboard log files. The folder name will
   be _{yyyymmdd-hhmmss}_.  These logs can be used to create a [tensorboard](https://www.tensorflow.org/tensorboard)
   visualization of the training results.

## Building a Dataset

The raw clips were scrubbed of all on-screen information
(e.g. vendor logos, battery indicators, index mark, depth markers)
extraneous to the ultrasound beam itself. This was done using a dedicated
deep learning masking software for ultrasound (AutoMask, WaveBase Inc.,
Waterloo, Canada). Following this, all ultrasound clips were deconstructed into
their constituent frames, and a frame table was generated linking each frame to
their ground truth, associated clip, and patient.

The following csv headers can be used to create a clips table csv file to train a model:  

_| filename | patient_id  | view  | class |_  

Where _filename_ is the name of the labeled clip file, _patient_id_ is a unique patient identifier, 
_view_ is a string label for the clip, and _class_ is the label as a class integer.  

Using this clips table csv, a set of lung ultrasound clips in mp4 format, and the 
[_build-dataset.py_](/src/data/build-dataset.py) script, a frames table can be generated 
that will be used to train a model.
   
## Use Cases

### Train a Model

With a pre-processed clip dataset, you can train a frame classification model of a chosen model definition.
1. Assemble in a pre-processed clip dataset (see [**_Building a Dataset_**](#building-a-dataset)) and set the appropriate data paths in [_config.yml_](config.yml). 
2. Mask the images of extraneous information outside the ultrasound beam. In our group, this was done with proprietary software mentioned above in 'Building a Dataset'
3. Generate a frame dataset from the masked clips using [_build-dataset.py_](/src/data/build-dataset.py).
4. Set the desired data and train configuration fields in [_config.yml_](config.yml) including setting a model type using the `MODEL_DEF` parameter and setting the `EXPERIMENT_TYPE` to _single_train_.
5. Set the associated hyperparameter values based on the chosen model definition.
6. Run [_train.py_](/src/train.py).
7. View all logs and trained weights in the [_results_](/results) directory.

Note: We found that the _cutoffvgg16_ model definition had the best performance on our internal data.

### K-Fold Cross Validation

With a pre-processed clip dataset, you can evaluate model performance using k-fold cross-validation.
1. Assemble in a pre-processed clip dataset (see [**_Building a Dataset_**](#building-a-dataset)) 
   and set the appropriate data paths in [_config.yml_](config.yml).
2. Generate a frame from the masked clips using [_build-dataset.py_](/src/data/build-dataset.py).
3. Set the desired data and train configuration fields in [_config.yml_](config.yml) including setting a model type using the `MODEL_DEF` parameter and setting the `EXPERIMENT_TYPE` to _cross_validation_.
4. Set the number of folds in the train section of [_config.yml_](config.yml).
5. Set the associated hyperparameter values based on the chosen model definition.
6. Run [_train.py_](/src/train.py).
7. View all logs and trained weights in the [_results_](/results) directory. The partitions from each fold can be found in the [_partitions_](/src/results/data/partitions) folder.

### Hyper Parameter Optimization 

With a pre-processed clip dataset, you can perform a hyperparameter search to assist with hyperparameter optimization.
1. Assemble in a pre-processed clip dataset (see [**_Building a Dataset_**](#building-a-dataset)) 
   and set the appropriate data paths in [_config.yml_](config.yml).
2. Generate a frame dataset from the masked clips using [_build-dataset.py_](/src/data/build-dataset.py).
3. Set the desired data and train configuration field in [_config.yml_](config.yml) including setting a model type using the `MODEL_DEF` parameter and setting the `EXPERIMENT_TYPE` to _hparam_search_.
4. Set the hyperparameter search fields in the train section of [_config.yml_](config.yml).
5. Set the associated hyperparameter search configuration values based on the chosen model definition.
6. Run [_train.py_](/src/train.py).
7. View all logs in the [_logs_](/results/logs) folder and view bayesian hyperparameter search results in the [_experiments_](/results/experiments) folder.

### Predictions

With a trained model, you can compute frame predictions and clip predictions using the following steps:
1. Set the `MODEL_TO_LOAD` field in [_config.yml_](config.yml) to point to a trained model (in `.h5` format).
2. Set the `FRAME_TABLE` and `CLIPS_TABLE` fields to the dataset of interest. Set the `FRAMES` field to point to the dataset's directory of LUS frames.
3. Set the `CLIP_PREDICTION` > `CLIP_PREDICTION_METHOD` field to determine which algorithm is used to compute clip-wise predictions, given the clip's set of frame predictions produced by the model. Below is a brief description of each algorithm available.
   - **"contiguous"**: If the number of contiguous frames for which the frame's predicted B-line probability meets or exceeds the classification threshold is at least the contiguity threshold, classify the clip as "B-lines".
    - **"average"**: Compute the average prediction probabilities across the entire clip. If the B-line average probability meets or exceeds the classification threshold, classify the clip as "B-lines".
    - **"sliding_window"**: Take the clip's B-line probability as the greatest average B-line probability present in any contiguous set of frames as large as the sliding window.
4. Execute [predict.py](/src/predict.py).
5. Access the frame and corresponding clip predictions as CSV files, located in [results/predictions](/results/predictions/).

### Grad-CAM for Individual Frame Predictions

With a trained model and a collection of frame data, you can apply a Grad-CAM visualization to individual frames.
1. Set the `MODEL_TO_LOAD` field in [_config.yml_](config.yml) to point to a trained model (in `.h5` format).
2. Set the `FRAME_TABLE` field and set the `FRAMES` path field to point to a directory of LUS frames .
3. Run [_gradcam.py_](/src/explainability/gradcam.py).
4. Select the frame that you want to apply Grad-CAM to.
5. View Grad-CAM results in the [_img/heatmaps_](/img/heatmaps) folder.

## Project Configuration
This project contains several configurable variables that are defined in
the project config file: [config.yml](config.yml). When loaded into
Python scripts, the contents of this file become a dictionary through
which the developer can easily access its members.

For user convenience, the config file is organized into major components
of the model development pipeline. Many fields need not be modified by
the typical user, but others may be modified to suit the user's specific
goals. A summary of the major configurable elements in this file is
below.
<details closed>

<summary>Paths</summary>

This section of the config contains all path definitions for reading data and writing outputs. Any not defined are
pointing to directories rather than files.
- **FRAMES_TABLE**: Path to frame csv.
- **FRAMES_DIR**: Path to clip csv.
- **CLIPS_TABLE**
- **PARTITIONS_DIR**
- **RAW_CLIPS_DIR**
- **MASKED_CLIPS_DIR**
- **DATABASE_QUERY**
- **TEST_FRAMES_TABLE**: Path to test frame csv.
- **TEST_CLIPS_TABLE**: Path to test clip csv.
- **FRAME_PREDICTIONS**: Path to pre-generated frame predictions.
- **EXPERIMENT_VISUALIZATIONS**
- **MODEL_TO_LOAD**: Trained model in h5 file format.
- **HEATMAPS**
- **LOGS**
- **IMAGES**
- **MODEL_WEIGHTS**
- **BATCH_PREDS**
- **METRICS**
- **EXPERIMENTS**
- **EXPERIMENT_IMG**
</details>

<details closed> 
<summary>Data</summary>

- **IMG_DIM**: Dimensions for frame resizing.
- **VAL_SPLIT**: Validation split.
- **TEST_SPLIT**: Test split.
- **CLASSES**: A string list of data classes.
</details>

<details closed> 
<summary>Train</summary>

- **MODEL_DEF**: Defines the type of frame model to train. One of {'vgg16', 'mobilenetv2', 'xception', 'efficientnetb7', 'custom_resnetv2', 'cutoffvgg16'}
- **N_CLASSES**: Number of classes/labels.
- **BATCH_SIZE**: Batch size.
- **EPOCHS**: Number of epocs.
- **PATIENCE**: Number of epochs with no improvement after which training will be stopped.
- **EXPERIMENT_TYPE**: Toggle mixed precision training. Necessary for training with Tensor Cores.
- **N_FOLDS**: Cross-validation folds.
- **DATA_AUG**: Data augmentation parameters.
  - **ZOOM_RANGE**
  - **HORIZONTAL_FLIP**
  - **WIDTH_SHIFT_RANGE**
  - **HEIGHT_SHIFT_RANGE**
  - **SHEAR_RANGE**
  - **ROTATION_RANGE**
  - **BRIGHTNESS_RANGE**
  - **CONTRAST_RANGE**
- **MEMORY_LIMIT**:Virtual device memory limit
</details>

<details closed>
<summary>Hyperparameters</summary>

Each model type has a list of configurable hyperparameters defined here.
- **MODELNAME1**
  - **LR**
  - **DROPOUT**
  - **L2_LAMBDA**
  - **NODES_FC0**
  - **FROZEN_LAYERS**
- **MODELNAME2**
  - **LR**
  - **DROPOUT**
  - **L2_LAMBDA**
  - **NODES_FC0**
  - **FROZEN_LAYERS**    
</details>

<details closed> 
<summary>Clip Prediction</summary>

This section contains values specific to predicting clips using a set of frame predictions.
- **CLASSIFICATION_THRESHOLD**: Threshold for pleural classification.
- **CLIP_PREDICTION_METHOD**: Clip prediction logic. See  [config.yml](config.yml) for list of options.
- **WINDOW_SIZE**: Window size for sliding window clip predictions.
- **CONTIGUITY_THRESHOLD**: Threshold for contiguity threshold method.
- **SMOOTHING_WINDOW**: Smoothing window for smoothed contiguity threshold method.
- **WINDOW_CERTAINTY**: Certainly value for longest window method.
</details>

<details closed> 
<summary>Hyperparameter Search</summary>


For each model there is a range of values that can be sampled for the hyperparameter search.
The ranges are defined here in the config file. Each hyperparameter has a name, type, and range.
The type dictates how samples are drawn from the range.

For more information on using bayesian hyperparameters, visit the [skopt documentation](https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html).
- **OBJECTIVE** Hyperparameter search maximization/minimization metric.
- **N_TRIALS** Number of trials in search.
- **MODELNAME**
  - **HYPERPARAMETER_NAME1**
    - **TYPE**
    - **RANGE**
  - **HYPERPARAMETER_NAME2**
    - **TYPE**
    - **RANGE**

</details>

## Project Structure
The project looks similar to the directory structure below. Disregard
any _.gitkeep_ files, as their only purpose is to force Git to track
empty directories. Disregard any _.\__init\__.py_ files, as they are
empty files that enable Python to recognize certain directories as
packages.

```
├── img
|   ├── experiments                  <- Visualizations for experiments
|   ├── heatmaps                     <- Grad-CAM heatmap images
|   └── readme                       <- Image assets for README.md
├── results
|   ├── data                         
|   |   └── partition                <- K-fold cross-validation fold partitions
|   ├── experiments                  <- Experiment results
|   ├── figures                      <- Generated figures
|   ├── metrics                      <- Metrics for frame and clip inference
|   ├── models                       <- Trained model output
|   ├── predictions                  <- Prediction output
│   └── logs                         <- TensorBoard logs
├── src
│   ├── data
|   |   ├── build-dataset.py         <- Builds a table of frame examples using a table of clip metadata
|   |   ├── preprocessor.py          <- For preprocessing images for training and inference
|   |   ├── database_pull.py         <- Script for pulling clip mp4 files from the cloud - step 2 (specific to our setup)
|   |   └── query_to_df.py           <- Script for pulling clip metadata from the cloud - step 1 (specific to our setup)
│   ├── explainability
|   |   └── gradcam.py               <- Script containing some explainability method
│   ├── models
|   |   └── models.py                <- Script containing all model definitions
│   ├── visualization
|   |   └── visualization.py         <- Script for visualization production
|   ├── predict.py                   <- Script for prediction on raw data using trained models
|   └── train.py                     <- Script for training experiments
├── .gitignore                       <- Files to be be ignored by git.
├── config.yml                       <- Values of several constants used throughout project
├── README.md                        <- Project description
└── requirements.txt                 <- Lists all dependencies and their respective versions
```

## Contacts

**Robert Arntfield**  
Project Lead  
Deep Breathe  
robert.arntfield@gmail.com

**Bennett VanBerlo**  
Machine Learning Developer 
Deep Breathe  
bennettjlvb@gmail.com

**Blake VanBerlo**  
Deep Learning Project Lead   
Deep Breathe  
bvanberlo@uwaterloo.ca

