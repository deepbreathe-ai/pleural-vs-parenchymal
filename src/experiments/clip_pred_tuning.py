import os

import matplotlib
import yaml
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import multiprocessing

from src.predict import max_contiguous_pleural_preds, max_sliding_window, compute_metrics, \
    longest_window, majority_vote, avg_clip_prediction
from src.visualization.visualization import plot_clip_pred_experiment

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))


def explore_contiguity_threshold(frame_preds, class_thresh=0.5, min_contiguity_thresh=5, max_contiguity_thresh=60):
    '''
    Acquires a clip-wise accuracy score for a set of frame predictions across a range of contiguity thresholds.
    :frame_preds: DataFrame containing frame prediction probabilities
    :class_thresh: Classification threshold (in [0, 1]) used for all prediction generation
    :min_tau: Minimum contiguity threshold for experiment
    :max_tau: Maximum contiguity threshold for experiment
    '''
    frames_table_path = cfg['PATHS']['TEST_FRAMES_TABLE']
    clips_table_path = cfg['PATHS']['TEST_CLIPS_TABLE']

    frames_df = pd.read_csv(frames_table_path)
    clips_df = pd.read_csv(clips_table_path)

    clip_names = clips_df['filename']
    clip_labels = clips_df['Class']

    accuracies = []
    contiguity_thresholds = np.arange(min_contiguity_thresh, max_contiguity_thresh, 5)

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
    print(accuracies)
    metrics_df = pd.DataFrame({'tau': contiguity_thresholds, 'accuracy': accuracies})
    print(metrics_df)
    plot_clip_pred_experiment(metrics_df=metrics_df,
                              var_col='tau',
                              metrics_to_plot=['accuracy'],
                              im_path=cfg['PATHS']['EXPERIMENT_IMG'],
                              title='Accuracy vs Contiguity Threshold With Classification Threshold ' + str(
                                  class_thresh),
                              x_label='Contiguity Threshold',
                              experiment_type='contiguity_threshold')

def explore_sliding_window(frame_preds, min_window=5, max_window=60):
    '''
    Acquires a clip-wise accuracy score for a set of frame predictions across a range of contiguity thresholds.
    :frame_preds: DataFrame containing frame prediction probabilities
    :min_tau: Minimum window size for experiment
    :max_tau: Maximum window size for experiment
    '''
    frames_table_path = cfg['PATHS']['TEST_FRAMES_TABLE']
    clips_table_path = cfg['PATHS']['TEST_CLIPS_TABLE']

    frames_df = pd.read_csv(frames_table_path)
    clips_df = pd.read_csv(clips_table_path)

    clip_names = clips_df['filename']
    clip_labels = clips_df['Class']

    accuracies = []
    windows = np.arange(min_window, max_window, 5)

    # for each window size, acquire an accuracy score
    for window in windows:
        clip_pred_classes = []
        print("Making predictions using window = {}".format(window))
        # obtain class predictions using the current window
        for i in tqdm(range(len(clip_names)), position=0, leave=True):
            clip_name = clip_names[i]
            clip_files_df = frames_df[frames_df['Frame Path'].str.contains(clip_name)]
            if clip_files_df.shape[0] == 0:
                raise Exception("Clip {} had 0 frames. Aborting.".format(clip_name))
            clip_frame_preds = frame_preds[frame_preds['Frame Path'].str.contains(clip_name)]
            pred_probs = clip_frame_preds['Pleural View Probability'].to_numpy()
            clip_pred_class = max_sliding_window(pred_probs, window_size=window)
            clip_pred_classes.append(clip_pred_class)
        metrics = compute_metrics(np.array(clip_labels), np.array(clip_pred_classes))
        accuracies.append(metrics['accuracy'])
    print(accuracies)
    metrics_df = pd.DataFrame({'window': windows, 'accuracy': accuracies})
    print(metrics_df)
    plot_clip_pred_experiment(metrics_df=metrics_df,
                              var_col='window',
                              metrics_to_plot=['accuracy'],
                              im_path=cfg['PATHS']['EXPERIMENT_IMG'],
                              title='Accuracy vs Window Size',
                              x_label='Window Size',
                              experiment_type='sliding-window')

def explore_longest_window(frame_preds, min_certainty=0.05, max_certainty = 0.4):
    '''
    Acquires a clip-wise accuracy score for a set of frame predictions across a range of contiguity thresholds.
    :frame_preds: DataFrame containing frame prediction probabilities
    :min_tau: Minimum window size for experiment
    :max_tau: Maximum window size for experiment
    '''
    frames_table_path = cfg['PATHS']['TEST_FRAMES_TABLE']
    clips_table_path = cfg['PATHS']['TEST_CLIPS_TABLE']

    frames_df = pd.read_csv(frames_table_path)
    clips_df = pd.read_csv(clips_table_path)

    clip_names = clips_df['filename']
    clip_labels = clips_df['Class']

    accuracies = []
    certainties = np.arange(min_certainty, max_certainty, 0.05)

    # for each certainty value, acquire an accuracy score
    for certainty in certainties:
        clip_pred_classes = []
        print("Making predictions using certainty = {}".format(certainty))
        # obtain class predictions using the current certainty value
        for i in tqdm(range(len(clip_names)), position=0, leave=True):
            clip_name = clip_names[i]
            clip_files_df = frames_df[frames_df['Frame Path'].str.contains(clip_name)]
            if clip_files_df.shape[0] == 0:
                raise Exception("Clip {} had 0 frames. Aborting.".format(clip_name))
            clip_frame_preds = frame_preds[frame_preds['Frame Path'].str.contains(clip_name)]
            pred_probs = clip_frame_preds['Pleural View Probability'].to_numpy()
            clip_pred_class = longest_window(pred_probs, window_certainty=certainty, class_thresh=cfg['CLIP_PREDICTION']['CLASSIFICATION_THRESHOLD'])
            clip_pred_classes.append(clip_pred_class)
        metrics = compute_metrics(np.array(clip_labels), np.array(clip_pred_classes))
        accuracies.append(metrics['accuracy'])
    print(accuracies)
    metrics_df = pd.DataFrame({'certainty': certainties, 'accuracy': accuracies})
    print(metrics_df)
    plot_clip_pred_experiment(metrics_df=metrics_df,
                              var_col='certainty',
                              metrics_to_plot=['accuracy'],
                              im_path=cfg['PATHS']['EXPERIMENT_IMG'],
                              title='Accuracy vs Certainty Value',
                              x_label='Window Size',
                              experiment_type='longest-window')

def report_constant_clip_metrics(frame_preds):
    '''
    Plots accuracy of clip prediction algorithms that do not have variables to explore.
    (Including majority vote and average probability)
    :param frame_preds: DataFrame containing frame prediction probabilities
    '''
    frames_table_path = cfg['PATHS']['TEST_FRAMES_TABLE']
    clips_table_path = cfg['PATHS']['TEST_CLIPS_TABLE']

    frames_df = pd.read_csv(frames_table_path)
    clips_df = pd.read_csv(clips_table_path)

    clip_names = clips_df['filename']
    clip_labels = clips_df['Class']
    clip_pred_classes_avg = []
    clip_pred_classes_majority = []
    for i in tqdm(range(len(clip_names)), position=0, leave=True):
        clip_name = clip_names[i]
        clip_files_df = frames_df[frames_df['Frame Path'].str.contains(clip_name)]
        if clip_files_df.shape[0] == 0:
            raise Exception("Clip {} had 0 frames. Aborting.".format(clip_name))
        clip_frame_preds = frame_preds[frame_preds['Frame Path'].str.contains(clip_name)]
        pred_probs = clip_frame_preds['Pleural View Probability'].to_numpy()
        clip_pred_classes_avg.append(avg_clip_prediction(pred_probs, cfg['CLIP_PREDICTION']['CLASSIFICATION_THRESHOLD'])[0])
        clip_pred_classes_majority.append(majority_vote(pred_probs, cfg['CLIP_PREDICTION']['CLASSIFICATION_THRESHOLD']))
    metrics_avg = compute_metrics(np.array(clip_labels), np.array(clip_pred_classes_avg))
    metrics_majority = compute_metrics(np.array(clip_labels), np.array(clip_pred_classes_majority))
    print("Average Clip Prediction Accuracy: {}".format(metrics_avg['accuracy']))
    print("Majority Vote Clip Prediction Accuracy: {}".format(metrics_majority['accuracy']))


def full_experiment(frame_preds, experiment_cfg):
    explore_longest_window(frame_preds,
                           experiment_cfg['LONGEST_WINDOW_CERTAINTIES'][0],
                           experiment_cfg['LONGEST_WINDOW_CERTAINTIES'][1])
    explore_sliding_window(frame_preds,
                           experiment_cfg['SLIDING_WINDOW_LENGTHS'][0],
                           experiment_cfg['SLIDING_WINDOW_LENGTHS'][1])
    explore_contiguity_threshold(frame_preds,
                                 experiment_cfg['CONTIGUITY_THRESHOLD_CLASS_THRESH'],
                                 experiment_cfg['CONTIGUITY_THRESHOLD_RANGE'][0],
                                 experiment_cfg['CONTIGUITY_THRESHOLD_RANGE'][1])
    report_constant_clip_metrics(frame_preds)

if __name__ == '__main__':
    frame_preds = pd.read_csv(cfg['PATHS']['FRAME_PREDICTIONS'])
    # run an experiment on all clip prediction algorithms
    full_experiment(frame_preds, cfg['CLIP_PREDICTION']['EXPERIMENTS'])
