import json
import os
from datetime import datetime

import matplotlib
import yaml
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import multiprocessing
from tensorflow.keras.models import load_model

from models.models import get_model
from src.predict import max_contiguous_pleural_preds, max_sliding_window, compute_metrics, \
    longest_window, majority_vote, avg_clip_prediction, predict_set
from src.visualization.visualization import plot_clip_pred_experiment

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

def explore_contiguity_threshold(frame_preds, class_thresh=0.5, min_contiguity_thresh=1, max_contiguity_thresh=50, i = 0, metrics = True):
    '''
    Acquires a clip-wise accuracy score for a set of frame predictions across a range of contiguity thresholds.
    :frame_preds: DataFrame containing frame prediction probabilities
    :class_thresh: Classification threshold (in [0, 1]) used for all prediction generation
    :min_tau: Minimum contiguity threshold for experiment
    :max_tau: Maximum contiguity threshold for experiment
    '''
    frames_table_path = 'C:/Users/Bennet/DeepBreathe/kfold_metrics/kfold20220224-093231_partitions/kfold20220224-093231/fold_{}_val_set.csv'.format(i)
    clips_table_path = 'C:/Users/Bennet/DeepBreathe/kfold_metrics/kfold20220224-093231_partitions/kfold20220224-093231/fold_{}_val_clips.csv'.format(i)

    frames_df = pd.read_csv(frames_table_path)
    clips_df = pd.read_csv(clips_table_path)

    clip_names = clips_df['filename']
    clip_labels = clips_df['Class']

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    contiguity_thresholds = np.arange(min_contiguity_thresh, max_contiguity_thresh, 1)

    # for each contiguity threshold, acquire an accuracy score
    for tau in contiguity_thresholds:
        clip_pred_classes = []
        print("Making predictions using tau = {}".format(tau))
        # obtain class predictions using the current tau
        for i in tqdm(range(len(clip_names)), position=0, leave=True):
            clip_name = clip_names[i]
            clip_files_df = frames_df[frames_df['Frame Path'].str.contains(clip_name)]
            if clip_files_df.shape[0] == 0:
                raise Exception("Clip {} had 0 frames. Aborting.".format(clip_name))
            clip_frame_preds = frame_preds[frame_preds['Frame Path'].str.contains(clip_name)]
            pred_probs = clip_frame_preds['Pleural View Probability'].to_numpy()
            clip_pred_class = max_contiguous_pleural_preds(pred_probs, class_thresh, tau)
            clip_pred_classes.append(clip_pred_class)
        metrics = compute_metrics(np.array(clip_labels), np.array(clip_pred_classes))
        accuracies.append(metrics['accuracy'])
        if(metrics):
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1s.append(metrics['f1'])

    print(accuracies)
    metrics_df = pd.DataFrame({'tau': contiguity_thresholds, 'accuracy': accuracies, 'precision': precisions,
                               'recall': recalls, 'f1': f1s})
    print(metrics_df)
    return accuracies

def explore_majority_vote(frame_preds, class_thresh=0.5, i = 0, metrics = True):
    '''
        Acquires a clip-wise accuracy score for a set of frame predictions across a range of contiguity thresholds.
        :frame_preds: DataFrame containing frame prediction probabilities
        :class_thresh: Classification threshold (in [0, 1]) used for all prediction generation
        :min_tau: Minimum contiguity threshold for experiment
        :max_tau: Maximum contiguity threshold for experiment
        '''
    frames_table_path = 'C:/Users/Bennet/DeepBreathe/kfold_metrics/kfold20220224-093231_partitions/kfold20220224-093231/fold_{}_val_set.csv'.format(
        i)
    clips_table_path = 'C:/Users/Bennet/DeepBreathe/kfold_metrics/kfold20220224-093231_partitions/kfold20220224-093231/fold_{}_val_clips.csv'.format(
        i)

    frames_df = pd.read_csv(frames_table_path)
    clips_df = pd.read_csv(clips_table_path)

    clip_names = clips_df['filename']
    clip_labels = clips_df['Class']

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    # for each contiguity threshold, acquire an accuracy score
    clip_pred_classes = []
    print("Making predictions for fold {}".format(i))
    # obtain class predictions using the current tau
    for j in tqdm(range(len(clip_names)), position=0, leave=True):
        clip_name = clip_names[j]
        clip_files_df = frames_df[frames_df['Frame Path'].str.contains(clip_name)]
        if clip_files_df.shape[0] == 0:
            raise Exception("Clip {} had 0 frames. Aborting.".format(clip_name))
        clip_frame_preds = frame_preds[frame_preds['Frame Path'].str.contains(clip_name)]
        pred_probs = clip_frame_preds['Pleural View Probability'].to_numpy()
        clip_pred_class = majority_vote(pred_probs, class_thresh)
        clip_pred_classes.append(clip_pred_class)
    metrics = compute_metrics(np.array(clip_labels), np.array(clip_pred_classes))
    accuracies.append(metrics['accuracy'])
    if (metrics):
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1s.append(metrics['f1'])

    print(accuracies)
    metrics_df = pd.DataFrame({'accuracy': accuracies, 'precision': precisions,
                               'recall': recalls, 'f1': f1s})
    print(metrics_df)
    return accuracies

def explore_average_clip_pred(frame_preds, class_thresh=0.5, i = 0, metrics = True):
    '''
        Acquires a clip-wise accuracy score for a set of frame predictions across a range of contiguity thresholds.
        :frame_preds: DataFrame containing frame prediction probabilities
        :class_thresh: Classification threshold (in [0, 1]) used for all prediction generation
        :min_tau: Minimum contiguity threshold for experiment
        :max_tau: Maximum contiguity threshold for experiment
        '''
    frames_table_path = 'C:/Users/Bennet/DeepBreathe/kfold_metrics/kfold20220224-093231_partitions/kfold20220224-093231/fold_{}_val_set.csv'.format(
        i)
    clips_table_path = 'C:/Users/Bennet/DeepBreathe/kfold_metrics/kfold20220224-093231_partitions/kfold20220224-093231/fold_{}_val_clips.csv'.format(
        i)

    frames_df = pd.read_csv(frames_table_path)
    clips_df = pd.read_csv(clips_table_path)

    clip_names = clips_df['filename']
    clip_labels = clips_df['Class']

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    # for each contiguity threshold, acquire an accuracy score
    clip_pred_classes = []
    print("Making predictions for fold {}".format(i))
    # obtain class predictions using the current tau
    for j in tqdm(range(len(clip_names)), position=0, leave=True):
        clip_name = clip_names[j]
        clip_files_df = frames_df[frames_df['Frame Path'].str.contains(clip_name)]
        if clip_files_df.shape[0] == 0:
            raise Exception("Clip {} had 0 frames. Aborting.".format(clip_name))
        clip_frame_preds = frame_preds[frame_preds['Frame Path'].str.contains(clip_name)]
        pred_probs = clip_frame_preds['Pleural View Probability'].to_numpy()
        clip_pred_class = avg_clip_prediction(pred_probs, class_thresh)
        clip_pred_classes.append(clip_pred_class[0])
    metrics = compute_metrics(np.array(clip_labels), np.array(clip_pred_classes))
    accuracies.append(metrics['accuracy'])
    if (metrics):
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1s.append(metrics['f1'])

    print(accuracies)
    metrics_df = pd.DataFrame({'accuracy': accuracies, 'precision': precisions,
                               'recall': recalls, 'f1': f1s})
    print(metrics_df)
    return accuracies

def compute_frame_predictions(cfg, dataset_files_path, class_thresh=0.5, calculate_metrics=True, i=0):
    '''
    For a particular dataset, make predictions for each image and compute metrics. Save the resultant metrics.
    :param cfg: project config
    :param dataset_files_path: Path to CSV of Dataframe linking filenames to labels
    :param class_thresh: Classification threshold for frame prediction
    :param calculate_metrics: If True, calculate metrics for these predictions; if so, ensure you have a ground truth column
    '''
    model_type = cfg['TRAIN']['MODEL_DEF']
    _, preprocessing_fn = get_model(model_type)
    print("Loading {}".format('fold{}.h5'.format(i)))
    model = load_model('C:/Users/Bennet/DeepBreathe/kfold_metrics/kfold_20220224-093230_models/kfold_20220224-093230/fold{}.h5'.format(i), compile=False)
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
    # pred_probs_df.to_csv(os.path.join(cfg['PATHS']['BATCH_PREDS'], 'frame_predictions' +
    #                                   datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv'))
    return pred_probs_df

if __name__ == '__main__':
    df = pd.DataFrame()
    for i in range(0,10):
        print("Computing Frame preds for fold {}".format(i))
        frames_path = 'C:/Users/Bennet/DeepBreathe/kfold_metrics/kfold20220224-093231_partitions/kfold20220224-093231/fold_{}_val_set.csv'.format(i)
        frame_preds = compute_frame_predictions(cfg, frames_path, class_thresh=0.5, calculate_metrics=False, i=i)
        print("Exploring contiguity thresh for fold {}".format(i))
        col_name = 'fold{}'.format(i)
        df[col_name] = explore_average_clip_pred(frame_preds, 0.5, i)
        #df[col_name] = explore_contiguity_threshold(frame_preds, 0.5, 19, 20, i = i, metrics = True)
    for x in df:
        print(x)
    df.to_csv(os.path.join(cfg['PATHS']['EXPERIMENTS'], 'k_fold_experiment_predictions_tau19.csv'))

