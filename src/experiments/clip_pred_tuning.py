import os

import yaml
import pandas as pd
import numpy as np


from src.predict import max_contiguous_pleural_preds, max_sliding_window, compute_metrics

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

def explore_contiguity_threshold(frame_preds, class_thresh=0.5, min_tau=5, max_tau=60):
    frames_table_path = cfg['PATHS']['TEST_FRAMES_TABLE']
    clips_table_path = cfg['PATHS']['TEST_CLIPS_TABLE']
    frames_df = pd.read_csv(frames_table_path)
    clips_df = pd.read_csv(clips_table_path)
    clip_names = clips_df['filename']

    pred_classes = frame_preds['Class']
    pred_probs = frame_preds['Pleural View Probability']
    clip_labels = clips_df['Class']
    accuracies = []
    taus = np.arange(min_tau, max_tau, 5)

    for tau in taus:
        clip_pred_classes = []
        for i in range(len(clip_names)):
            clip_name = clip_names[i]
            clip_files_df = frames_df[frames_df['Frame Path'].str.contains(clip_name)]
            if clip_files_df.shape[0] == 0:
                raise Exception("Clip {} had 0 frames. Aborting.".format(clip_name))
            clip_frame_preds = frame_preds[frame_preds['Frame Path'].str.contains(clip_name)]
            pred_probs = clip_frame_preds['Pleural View Probability'].to_numpy()
            print("Making predictions for clip {} with tau {}".format(clip_name, tau))
            clip_pred_class = max_contiguous_pleural_preds(pred_probs, class_thresh, tau)
            clip_pred_classes.append(clip_pred_class)
        metrics = compute_metrics(np.array(clip_labels), np.array(clip_pred_classes))
        accuracies.append(metrics['accuracy'])
    print(accuracies)
    metrics_df = pd.DataFrame({'tau': taus, 'accuracy': accuracies})
    print(metrics_df)

if __name__=='__main__':

    frame_preds = pd.read_csv(cfg['PATHS']['FRAME_PREDICTIONS'])
    explore_contiguity_threshold(frame_preds)