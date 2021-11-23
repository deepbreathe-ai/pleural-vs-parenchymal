import time
import json

import os
import numpy as np
import pandas as pd
from sklearn.metrics import *
from tensorflow.keras.models import load_model
import yaml

from src.visualization.visualization import *
from src.models.models import get_model
from src.data.preprocessor import Preprocessor

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))


def predict_set(model, preprocessing_fn, predict_df, threshold=0.5):
    '''
    Given a dataset, make predictions for each constituent example.
    :param model: A trained TensorFlow model
    :param preprocessing_fn: Preprocessing function to apply before sending image to model
    :param predict_df: Pandas Dataframe of LUS frames, linking image filenames to labels
    :param threshold: Classification threshold
    :return: List of predicted classes, array of classwise prediction probabilities
    '''

    # Create dataset from CSV of frames
    frames_dir = cfg['PATHS']['FRAMES_DIR']
    dataset = tf.data.Dataset.from_tensor_slices(([os.path.join(frames_dir, f) for f in predict_df['Frame Path'].tolist()], predict_df['Class']))
    preprocessor = Preprocessor(preprocessing_fn)
    preprocessed_set = preprocessor.prepare(dataset, shuffle=False, augment=False)

    # Obtain prediction probabilities
    p = model.predict(preprocessed_set)
    test_predictions = (p[:, 0] >= threshold).astype(int)

    # Get prediction classes in original labelling system
    pred_classes = [cfg['DATA']['CLASSES'][v] for v in list(test_predictions)]
    test_predictions = [cfg['DATA']['CLASSES'].index(c) for c in pred_classes]
    return test_predictions, p

def compute_metrics(cfg, labels, preds, probs=None):
    '''
    Given labels and predictions, compute some common performance metrics
    :param cfg: project config
    :param labels: List of labels
    :param preds: List of predicted classes
    :param probs: Array of predicted classwise probabilities
    :return: A dictionary of metrics
    '''

    metrics = {}
    class_names = cfg['DATA']['CLASSES']

    precision = precision_score(labels, preds, average='binary')
    recalls = recall_score(labels, preds, average=None)
    f1 = f1_score(labels, preds, average='binary')

    metrics['confusion_matrix'] = confusion_matrix(labels, preds).tolist()
    metrics['precision'] = precision
    metrics['recall'] = recalls[1]          # Recall of the positive class (i.e. sensitivity)
    metrics['specificity'] = recalls[0]     # Specificity is recall of the negative class
    metrics['f1'] = f1
    metrics['accuracy'] = accuracy_score(labels, preds)

    if probs is not None:
        metrics['macro_mean_auc'] = roc_auc_score(labels, probs[:,1], average='macro', multi_class='ovr')
        metrics['weighted_mean_auc'] = roc_auc_score(labels, probs[:,1], average='weighted', multi_class='ovr')

        # Calculate classwise AUCs
        for class_name in class_names:
            classwise_labels = (labels == class_names.index(class_name)).astype(int)
            class_probs = probs[:,class_names.index(class_name)]
            metrics[class_name + '_auc'] = roc_auc_score(classwise_labels, class_probs)
    return metrics


def compute_clip_predictions(cfg, frames_table_path, clips_table_path, class_thresh=0.5, calculate_metrics=True):
    '''
    For a particular dataset, use predictions for each filename to create predictions for whole clips and save the
    resulting metrics.
    :param cfg: project config
    :param frames_table_path: Path to CSV of Dataframe linking filenames to labels
    :param clips_table_path: Path to CSV of Dataframe linking clips to labels
    :param class_thresh: Classification threshold for frame prediction
    :param clip_algorithm: Choice of clip prediction algorithm (one of 'contiguous', 'average', 'sliding_window')
    :param calculate_metrics: If True, calculate metrics for these predictions; if so, ensure you have a ground truth column
    '''
    model_type = cfg['TRAIN']['MODEL_DEF']
    _, preprocessing_fn = get_model(model_type)
    model = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)
    set_name = frames_table_path.split('/')[-1].split('.')[0] + '_clips'

    frames_df = pd.read_csv(frames_table_path)
    clips_df = pd.read_csv(clips_table_path)

    clip_names = clips_df['filename']
    clip_pred_classes = []
    all_pred_probs = np.zeros((clips_df.shape[0], len(cfg['DATA']['CLASSES'])))
    print("Found {} clips. Determining clip predictions.".format(clips_df.shape[0]))
    for i in range(len(clip_names)):

        # Get records from all files from this clip
        clip_name = clip_names[i]
        clip_files_df = frames_df[frames_df['Frame Path'].str.contains(clip_name)]
        print("Making predictions for clip {} with {} frames".format(clip_name, clip_files_df.shape[0]))
        if clip_files_df.shape[0] == 0:
            raise Exception("Clip {} had 0 frames. Aborting.".format(clip_name))

        # Make predictions for each image
        pred_classes, pred_probs = predict_set(model, preprocessing_fn, clip_files_df, threshold=class_thresh)

        # Compute average prediction for entire clip
        clip_pred_prob = np.mean(pred_probs, axis=0)
        all_pred_probs[i] = clip_pred_prob

        # Record predicted class
        pred_class = (clip_pred_prob[0] >= class_thresh).astype(int)
        clip_pred_classes.append(pred_class)

    if calculate_metrics:
        clip_labels = clips_df['class']
        metrics = compute_metrics(cfg, np.array(clip_labels), np.array(clip_pred_classes), all_pred_probs)
        json.dump(metrics, open(cfg['PATHS']['METRICS'] + set_name +
                                 datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.json', 'w'))

    # Save predictions
    pred_probs_df = pd.DataFrame(all_pred_probs, columns=cfg['DATA']['CLASSES'])
    pred_probs_df.insert(0, 'filename', clips_df['filename'])
    pred_probs_df.insert(1, 'class', clips_df['class'])
    pred_probs_df.to_csv(cfg['PATHS']['BATCH_PREDS'] + set_name + '_predictions' +
                             datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv')
    return pred_probs_df


def compute_frame_predictions(cfg, dataset_files_path, class_thresh=0.5, calculate_metrics=True):
    '''
    For a particular dataset, make predictions for each image and compute metrics. Save the resultant metrics.
    :param cfg: project config
    :param dataset_files_path: Path to CSV of Dataframe linking filenames to labels
    :param class_thresh: Classification threshold for frame prediction
    :param calculate_metrics: If True, calculate metrics for these predictions; if so, ensure you have a ground truth column
    '''
    model_type = cfg['TRAIN']['MODEL_DEF']
    _, preprocessing_fn = get_model(model_type)
    model = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)
    set_name = dataset_files_path.split('/')[-1].split('.')[0] + '_frames'

    files_df = pd.read_csv(dataset_files_path)

    # Make predictions for each image
    pred_classes, pred_probs = predict_set(model, preprocessing_fn, files_df, threshold=class_thresh)

    # Compute and save metrics
    if calculate_metrics:
        frame_labels = files_df['Class']  # Get ground truth
        metrics = compute_metrics(cfg, np.array(frame_labels), np.array(pred_classes), pred_probs)
        json.dump(metrics, open(cfg['PATHS']['METRICS'] + set_name +
                                datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.json', 'w'))

    # Save predictions
    pred_probs_df = pd.DataFrame(pred_probs, columns=cfg['DATA']['CLASSES'])
    pred_probs_df.insert(0, 'Frame Path', files_df['Frame Path'])
    pred_probs_df.insert(1, 'Class', files_df['Class'])
    pred_probs_df.to_csv(cfg['PATHS']['BATCH_PREDS'] + set_name + '_predictions' +
                          datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv')
    return pred_probs_df


def clock_avg_runtime(n_gpu_warmup_runs, n_experiment_runs):
    '''
    Measures the average inference time of a trained model. Executes a few warm-up runs, then measures the inference
    time of the model over a series of trials.
    :param n_gpu_warmup_runs: The number of inference runs to warm up the GPU
    :param n_experiment_runs: The number of inference runs to record
    :return: Average and standard deviation of the times of the recorded inference runs
    '''
    times = np.zeros((n_experiment_runs))
    img_dim = cfg['DATA']['IMG_DIM']
    model = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)
    for i in range(n_gpu_warmup_runs):
        x = tf.random.normal((1, img_dim[0], img_dim[1], 3))
        y = model(x)
    for i in range(n_experiment_runs):
        x = tf.random.normal((1, img_dim[0], img_dim[1], 3))
        t_start = time.time()
        y = model(x)
        times[i] = time.time() - t_start
    t_avg_ms = np.mean(times) * 1000
    t_std_ms = np.std(times) * 1000
    print("Average runtime = {:.3f} ms, standard deviation = {:.3f} ms".format(t_avg_ms, t_std_ms))

if __name__ == '__main__':
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    frames_path = cfg['PATHS']['FRAMES_TABLE']
    clips_path = cfg['PATHS']['CLIPS_TABLE']
    compute_clip_predictions(cfg, frames_path, clips_path, class_thresh=cfg['CLIP_PREDICTION']['CLASSIFICATION_THRESHOLD'],
                             calculate_metrics=True)
    #compute_frame_predictions(cfg, frames_path, class_thresh=0.9, calculate_metrics=True)