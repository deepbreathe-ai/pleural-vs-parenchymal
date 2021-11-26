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
    return test_predictions, np.squeeze(p, axis=1)

def compute_metrics(labels, preds, probs=None):
    '''
    Given labels and predictions, compute some common performance metrics
    :param labels: List of labels
    :param preds: List of predicted classes
    :param probs: Array of predicted classwise probabilities
    :return: A dictionary of metrics
    '''

    metrics = {}

    precision = precision_score(labels, preds)
    recalls = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    metrics['confusion_matrix'] = confusion_matrix(labels, preds).tolist()
    metrics['precision'] = precision
    metrics['recall'] = recalls
    metrics['f1'] = f1
    metrics['accuracy'] = accuracy_score(labels, preds)

    if probs is not None:
        metrics['auc'] = roc_auc_score(labels, probs)
    return metrics


def compute_clip_predictions(cfg, frames_table_path, clips_table_path, class_thresh=0.5, clip_pred_method='average',
                             calculate_metrics=True):
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

    frames_df = pd.read_csv(frames_table_path)
    clips_df = pd.read_csv(clips_table_path)

    clip_names = clips_df['filename']
    clip_pred_classes = []
    all_pred_probs = np.zeros((clips_df.shape[0])) if clip_pred_method == 'average' else None
    print("Found {} clips. Determining clip predictions with the {} method.".format(clips_df.shape[0], clip_pred_method))
    for i in range(len(clip_names)):

        # Get records from all files from this clip
        clip_name = clip_names[i]
        clip_files_df = frames_df[frames_df['Frame Path'].str.contains(clip_name)]
        print("Making predictions for clip {} with {} frames".format(clip_name, clip_files_df.shape[0]))
        if clip_files_df.shape[0] == 0:
            raise Exception("Clip {} had 0 frames. Aborting.".format(clip_name))

        # Make predictions for each frame
        pred_classes, pred_probs = predict_set(model, preprocessing_fn, clip_files_df, threshold=class_thresh)

        # Determine a clip-wise prediction
        if clip_pred_method == 'majority_vote':
            clip_pred_class = majority_vote(pred_probs, class_thresh)
        elif clip_pred_method == 'contiguity_threshold':
            contiguity_threshold = cfg['CLIP_PREDICTION']['CONTIGUITY_THRESHOLD']
            clip_pred_class = max_contiguous_pleural_preds(pred_probs, class_thresh, contiguity_threshold)
        elif clip_pred_method == 'max_sliding_window':
            window_size = cfg['CLIP_PREDICTION']['WINDOW_SIZE']
            clip_pred_class = max_sliding_window(pred_probs, window_size)
        else:
            clip_pred_class, clip_pred_prob = avg_clip_prediction(pred_probs, class_thresh)
            all_pred_probs[i] = clip_pred_prob
        clip_pred_classes.append(clip_pred_class)         # Record predicted class for the clips

    if calculate_metrics:
        clip_labels = clips_df['Class']
        metrics = compute_metrics(np.array(clip_labels), np.array(clip_pred_classes), all_pred_probs)
        json.dump(metrics, open(os.path.join(cfg['PATHS']['METRICS'], 'clips_' +
                                             datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.json'), 'w'))

    # Save predictions
    pred_df = pd.DataFrame(clip_pred_classes)
    if all_pred_probs is not None:
        pred_df['Pred probs'] = all_pred_probs
    pred_df.insert(0, 'filename', clips_df['filename'])
    pred_df.insert(1, 'Class', clips_df['Class'])
    pred_df.to_csv(os.path.join(cfg['PATHS']['BATCH_PREDS'] + 'clip_predictions' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv'))
    return pred_df


def avg_clip_prediction(pred_probs, class_thresh):
    '''
    Returns the average probability among a list of frame prediction probabilities and the corresponding predicted class
    :param pred_probs (np.array) [n_frames]: Framewise prediction probabilities
    :param class_thresh (float): Classification threshold (in [0, 1])
    :return (int, float): Clip class prediction, clip probability prediction
    '''
    clip_pred_prob = np.mean(pred_probs, axis=0)
    clip_pred_class = (clip_pred_prob >= class_thresh).astype(int)
    return clip_pred_class, clip_pred_prob


def majority_vote(pred_probs, class_thresh):
    '''
    Determines the predicted class for each framewise prediction and determines a clip prediction via majority voting.
    :param pred_probs (np.array) [n_frames]: Framewise prediction probabilities
    :param class_thresh (float): Classification threshold (in [0, 1])
    :return (int): Clip class prediction
    '''
    pred_classes = (pred_probs >= class_thresh).astype(int)
    return (np.sum(pred_classes) >= pred_classes.shape[0] / 2.).astype(np.int32)


def max_sliding_window(pred_probs, window_size):
    '''
    Predicts a clipwise class by determining which class has the least difference between the class score and any
    average framewise probability taken over a sliding framewise window.
    :param pred_probs (np.array) [n_frames]: Framewise prediction probabilities
    :return (int): Clip class prediction
    '''
    prob_extremes = [1., 0.]
    for i in range(pred_probs.shape[0] - window_size):
        window_prob = np.mean(pred_probs[i:i + window_size])
        if window_prob < prob_extremes[0]:
            prob_extremes[0] = window_prob
        if window_prob > prob_extremes[1]:
            prob_extremes[1] = window_prob
    return prob_extremes[0] > 1. - prob_extremes[1]


def max_contiguous_pleural_preds(pred_probs, class_thresh, contiguity_thresh):
    '''
    Predicts a clipwise class as pleural if there is at least 1 instance of contiguity_thresh contiguous frames for
    which the prediction is pleural.
    :param pred_probs (np.array) [n_frames]: Framewise prediction probabilities
    :param class_thresh (float): Classification threshold (in [0, 1])
    :param contiguity_thresh (int): Number of contiguous frames for which the prediction is pleural to return a pleural
                                    clipwise prediction
    :return (int): Clip class prediction
    '''
    max_contiguous = cur_contiguous = 0
    for i in range(pred_probs.shape[0]):
        if pred_probs[i] >= class_thresh:
            cur_contiguous += 1
        else:
            cur_contiguous = 0
        if cur_contiguous > max_contiguous:
            max_contiguous = cur_contiguous
    return max_contiguous >= contiguity_thresh


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
        metrics = compute_metrics(np.array(frame_labels), np.array(pred_classes), pred_probs)
        json.dump(metrics, open(os.path.join(cfg['PATHS']['METRICS'], 'frames_' +
                                             datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.json'), 'w'))

    # Save predictions
    pred_probs_df = pd.DataFrame(pred_probs, columns=['Pleural View Probability'])
    pred_probs_df.insert(0, 'Frame Path', files_df['Frame Path'])
    pred_probs_df.insert(1, 'Class', files_df['Class'])
    pred_probs_df.to_csv(os.path.join(cfg['PATHS']['BATCH_PREDS'], 'frame_predictions' +
                                      datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv'))
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
    frames_path = cfg['PATHS']['TEST_FRAMES_TABLE']
    clips_path = cfg['PATHS']['TEST_CLIPS_TABLE']
    compute_clip_predictions(cfg, frames_path, clips_path,
                             clip_pred_method=cfg['CLIP_PREDICTION']['CLIP_PREDICTION_METHOD'],
                             class_thresh=cfg['CLIP_PREDICTION']['CLASSIFICATION_THRESHOLD'], calculate_metrics=True)
    #compute_frame_predictions(cfg, frames_path, class_thresh=0.5, calculate_metrics=True)