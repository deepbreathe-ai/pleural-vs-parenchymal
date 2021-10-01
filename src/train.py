import os
import yaml
import datetime

import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.models import *
from src.visualization.visualization import *
from src.data.preprocessor import Preprocessor

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

# for device in tf.config.experimental.list_physical_devices("GPU"):
#     tf.config.experimental.set_memory_growth(device, True)

def get_class_weights(histogram):
    '''
    Computes weights for each class to be applied in the loss function during training.
    :param histogram: A list depicting the number of each item in different class
    :param class_multiplier: List of values to multiply the calculated class weights by. For further control of class weighting.
    :return: A dictionary containing weights for each class
    '''
    weights = [None] * len(histogram)
    for i in range(len(histogram)):
        weights[i] = (1.0 / len(histogram)) * sum(histogram) / histogram[i]
    class_weight = {i: weights[i] for i in range(len(histogram))}
    print("Class weights: ", class_weight)
    return class_weight


def define_callbacks(cfg):
    '''
    Defines a list of Keras callbacks to be applied to model training loop
    :param cfg: Project config object
    :return: list of Keras callbacks
    '''
    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=cfg['TRAIN']['PATIENCE'], mode='min',
                                   restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=cfg['TRAIN']['PATIENCE'] // 2 + 1, verbose=1,
                                  min_lr=1e-8, min_delta=0.0001)

    # class ClearMemory(Callback):
    #     def on_epoch_end(self, epoch, logs=None):
    #         gc.collect()
    #         k.clear_session()

    callbacks = [early_stopping, reduce_lr]

    return callbacks


def partition_dataset(val_split, test_split, save_dfs=True):
    '''
    Partition the frame_df into training, validation and test sets by patient ID
    :param val_split: Validation split (in range [0, 1])
    :param test_split: Test split (in range [0, 1])
    :param save_dfs: Flag indicating whether to save the splits
    :return: (Training DataFrame, validation DataFrame, test DataFrame)
    '''

    frame_df = pd.read_csv(cfg['PATHS']['FRAMES_TABLE'])
    all_pts = frame_df['Patient'].unique()  # Get list of patients
    relative_val_split = val_split / (1 - (test_split))
    trainval_pts, test_pts = train_test_split(all_pts, test_size=test_split)
    train_pts, val_pts = train_test_split(trainval_pts, test_size=relative_val_split)

    train_df = frame_df[frame_df['Patient'].isin(train_pts)]
    val_df = frame_df[frame_df['Patient'].isin(val_pts)]
    test_df = frame_df[frame_df['Patient'].isin(test_pts)]
    print('TRAIN/VAL/TEST SPLIT: [{}, {}, {}] frames, [{}, {}, {}] patients'
          .format(train_df.shape[0], val_df.shape[0], test_df.shape[0], train_pts.shape[0], val_pts.shape[0],
                  test_pts.shape[0]))

    if save_dfs:
        train_df.to_csv(os.path.join(cfg['PATHS']['PARTITIONS_DIR'], 'train_set.csv'))
        val_df.to_csv(os.path.join(cfg['PATHS']['PARTITIONS_DIR'], 'val_set.csv'))
        test_df.to_csv(os.path.join(cfg['PATHS']['PARTITIONS_DIR'], 'test_set.csv'))
    return train_df, val_df, test_df


def log_test_results(model, test_set, test_df, test_metrics, log_dir):
    '''
    Visualize performance of a trained model on the test set. Optionally save the model.
    :param model: A trained TensorFlow model
    :param test_set: A TensorFlow image generator for the test set
    :param test_metrics: Dict of test set performance metrics
    :param log_dir: Path to write TensorBoard logs
    '''

    # Visualization of test results
    test_predictions = model.predict(test_set, verbose=0)
    test_labels = test_df['Class'].to_numpy()
    plt = plot_roc(test_labels, test_predictions, list(range(len(cfg['DATA']['CLASSES']))))
    roc_img = plot_to_tensor()
    plt = plot_confusion_matrix(test_labels, test_predictions, list(range(len(cfg['DATA']['CLASSES']))))
    cm_img = plot_to_tensor()

    # Log test set results and plots in TensorBoard
    writer = tf.summary.create_file_writer(logdir=log_dir)

    # Create table of test set metrics
    test_summary_str = [['**Metric**','**Value**']]
    for metric in test_metrics:
        metric_values = test_metrics[metric]
        test_summary_str.append([metric, str(metric_values)])

    # Create table of model and train hyperparameters used in this experiment
    hparam_summary_str = [['**Hyperparameter**', '**Value**']]
    for key in cfg['TRAIN']:
        hparam_summary_str.append([key, str(cfg['TRAIN'][key])])
    for key in cfg['HPARAMS'][cfg['TRAIN']['MODEL_DEF'].upper()]:
        hparam_summary_str.append([key, str(cfg['HPARAMS'][cfg['TRAIN']['MODEL_DEF'].upper()][key])])

    # Write to TensorBoard logs
    with writer.as_default():
        tf.summary.text(name='Test set metrics', data=tf.convert_to_tensor(test_summary_str), step=0)
        tf.summary.text(name='Run hyperparameters', data=tf.convert_to_tensor(hparam_summary_str), step=0)
        tf.summary.image(name='ROC Curve (Test Set)', data=roc_img, step=0)
        tf.summary.image(name='Confusion Matrix (Test Set)', data=cm_img, step=0)
    return

def train_model(model_def, preprocessing_fn, train_df, val_df, test_df, hparams, save_weights=False, log_dir=None, verbose=True):
    '''
    :param model_def: Model definition function
    :param preprocessing_fn: Model-specific preprocessing function
    :param train_df: Training set of LUS frames
    :param val_df: Validation set of LUS frames
    :param test_df: Test set of LUS frames
    :param hparams: Dict of hyperparameters
    :param save_weights: Flag indicating whether to save the model's weights
    :param log_dir: TensorBoard logs directory
    :param verbose: Whether to print out all epoch details
    :return: (model, test_metrics, test_generator)
    '''

    # Create TF datasets for training, validation and test sets
    frames_dir = cfg['PATHS']['FRAMES_DIR']
    train_set = tf.data.Dataset.from_tensor_slices(([os.path.join(frames_dir, f) for f in train_df['Frame Path'].tolist()], train_df['Class']))
    val_set = tf.data.Dataset.from_tensor_slices(([os.path.join(frames_dir, f) for f in val_df['Frame Path'].tolist()], val_df['Class']))
    test_set = tf.data.Dataset.from_tensor_slices(([os.path.join(frames_dir, f) for f in test_df['Frame Path'].tolist()], test_df['Class']))

    # Set up preprocessing transformations to apply to each item in dataset
    preprocessor = Preprocessor(preprocessing_fn)
    train_set = preprocessor.prepare(train_set, shuffle=True, augment=True)
    val_set = preprocessor.prepare(val_set, shuffle=False, augment=False)
    test_set = preprocessor.prepare(test_set, shuffle=False, augment=False)

    # Get class weights based on prevalences
    histogram = np.bincount(train_df['Class'].to_numpy().astype(int))  # Get class distribution
    class_weight = get_class_weights(histogram)

    # Define performance metrics
    classes = cfg['DATA']['CLASSES']
    n_classes = len(classes)
    threshold = 1.0 / n_classes # Binary classification threshold for a class
    metrics = ['accuracy', AUC(name='auc'), F1Score(name='f1score', num_classes=n_classes)]
    metrics += [Precision(name='precision_' + classes[i], thresholds=threshold, class_id=i) for i in range(n_classes)]
    metrics += [Recall(name='recall_' + classes[i], thresholds=threshold, class_id=i) for i in range(n_classes)]

    print('Training distribution: ',
          ['Class ' + classes[i] + ': ' + str(histogram[i]) + '. '
           for i in range(len(histogram))])
    input_shape = cfg['DATA']['IMG_DIM'] + [3]

    # Compute output bias
    output_bias = np.log([histogram[i] / (np.sum(histogram) - histogram[i]) for i in range(histogram.shape[0])])

    # Define the model
    model = model_def(hparams, input_shape, metrics, cfg['TRAIN']['N_CLASSES'], output_bias=output_bias)

    # Set training callbacks.
    callbacks = define_callbacks(cfg)
    if log_dir is not None:
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard)

    # Train the model.
    history = model.fit(train_set, epochs=cfg['TRAIN']['EPOCHS'], validation_data=val_set, callbacks=callbacks,
                        verbose=verbose, class_weight=class_weight)

    # Save the model's weights
    if save_weights:
        model_path = cfg['PATHS']['MODEL_WEIGHTS'] + 'model' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5'
        save_model(model, model_path)  # Save the model's weights

    # Run the model on the test set and print the resulting performance metrics.
    test_results = model.evaluate(test_set, verbose=1)
    test_metrics = {}
    test_summary_str = [['**Metric**', '**Value**']]
    for metric, value in zip(model.metrics_names, test_results):
        test_metrics[metric] = value
        test_summary_str.append([metric, str(value)])
    if log_dir is not None:
        log_test_results(model, test_set, test_df, test_metrics, log_dir)
    return model, test_metrics, test_set


def train_single(hparams=None, save_weights=False, write_logs=False):
    '''
    Train a single model. Use the passed hyperparameters if possible; otherwise, use those in config.
    :param hparams: Dict of hyperparameters
    :param save_model: Flag indicating whether to save the model
    :param write_logs: Flag indicating whether to write any training logs to disk
    :return: Dictionary of test set performance metrics
    '''
    train_df, val_df, test_df = partition_dataset(cfg['DATA']['VAL_SPLIT'], cfg['DATA']['TEST_SPLIT'])
    model_def, preprocessing_fn = get_model(cfg['TRAIN']['MODEL_DEF'])
    if write_logs:
        log_dir = os.path.join(cfg['PATHS']['LOGS'], datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        log_dir = None

    # Specify hyperparameters if not already done
    if hparams is None:
        hparams = cfg['HPARAMS'][cfg['TRAIN']['MODEL_DEF'].upper()]

    # Train the model
    model, test_metrics, _ = train_model(model_def, preprocessing_fn, train_df, val_df, test_df, hparams,
                                         save_weights=save_weights, log_dir=log_dir)
    print('Test set metrics: ', test_metrics)
    return test_metrics, model


def train_experiment(experiment='single_train', save_weights=False, write_logs=False):
    '''
    Run a training experiment
    :param experiment: String defining which experiment to run
    :param save_weights: Flag indicating whether to save any models trained during the experiment
    :param write_logs: Flag indicating whether to write logs for training
    '''

    # Conduct the desired train experiment
    if experiment == 'single_train':
        train_single(save_weights=save_weights, write_logs=write_logs)
    else:
        raise Exception("Invalid entry in TRAIN > EXPERIMENT_TYPE field of config.yml.")
    return


if __name__=='__main__':
    train_experiment(cfg['TRAIN']['EXPERIMENT_TYPE'], write_logs=True, save_weights=True)