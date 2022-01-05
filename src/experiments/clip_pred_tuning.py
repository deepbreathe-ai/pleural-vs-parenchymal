import os

import matplotlib
import yaml
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import multiprocessing



from src.predict import max_contiguous_pleural_preds, max_sliding_window, compute_metrics
from visualization.visualization import plot_clip_pred_experiment

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

def explore_contiguity_threshold(frame_preds, class_thresh=0.5, min_tau=5, max_tau=60):
    frames_table_path = cfg['PATHS']['TEST_FRAMES_TABLE']
    clips_table_path = cfg['PATHS']['TEST_CLIPS_TABLE']

    frames_df = pd.read_csv(frames_table_path)
    clips_df = pd.read_csv(clips_table_path)

    clip_names = clips_df['filename']
    clip_labels = clips_df['Class']

    accuracies = []
    taus = np.arange(min_tau, max_tau, 5)

    for tau in taus:
        clip_pred_classes = []
        print("Making predictions using tau = {}".format(tau))
        for i in tqdm(range(len(clip_names)),position=0, leave=True):
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
    metrics_df = pd.DataFrame({'tau': taus, 'accuracy': accuracies})
    print(metrics_df)
    plot_clip_pred_experiment(metrics_df=metrics_df,
                                        var_col='tau',
                                        metrics_to_plot=['accuracy'],
                                        im_path=cfg['PATHS']['EXPERIMENT_IMG'],
                                        title='Accuracy vs Contiguity Threshold With Classification Threshold '+str(class_thresh),
                                        x_label='Contiguity Threshold',
                                        experiment_type='contiguity_threshold')


def explore_sliding_window(frame_preds, min_window=5, max_window=60):
    frames_table_path = cfg['PATHS']['TEST_FRAMES_TABLE']
    clips_table_path = cfg['PATHS']['TEST_CLIPS_TABLE']

    frames_df = pd.read_csv(frames_table_path)
    clips_df = pd.read_csv(clips_table_path)

    clip_names = clips_df['filename']
    clip_labels = clips_df['Class']

    accuracies = []
    windows = np.arange(min_window, max_window, 5)

    for window in windows:
        clip_pred_classes = []
        print("Making predictions using window = {}".format(window))
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
    plot_clip_pred_experiment(metrics_df = metrics_df,
                                        var_col='window',
                                        metrics_to_plot=['accuracy'],
                                        im_path=cfg['PATHS']['EXPERIMENT_IMG'],
                                        title='Accuracy vs Window Size',
                                        x_label='Window Size',
                                        experiment_type = 'sliding-window')

if __name__=='__main__':
    frame_preds = pd.read_csv(cfg['PATHS']['FRAME_PREDICTIONS'])
    explore_sliding_window(frame_preds,5,60)
    explore_contiguity_threshold(frame_preds,0.5,5,60)
