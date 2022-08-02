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


def predict_set(model, preprocessing_fn, predict_df, threshold=0.5,fold=None):
    '''
    Given a dataset, make predictions for each constituent example.
    :param model: A trained TensorFlow model
    :param preprocessing_fn: Preprocessing function to apply before sending image to model
    :param predict_df: Pandas Dataframe of LUS frames, linking image filenames to labels
    :param threshold: Classification threshold
    :param fold: If not None, the cross-validation fold (0-9) to predict on
    :return: List of predicted classes, array of classwise prediction probabilities
    '''

    # Create dataset from CSV of frames
    if fold is not None:
        frames_dir = cfg['PATHS']['FRAMES_DIR'] + 'fold{}'.format(fold)
    else:
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
    cm = confusion_matrix(labels, preds).tolist()
    metrics['confusion_matrix'] = cm
    metrics['precision'] = precision
    metrics['recall'] = recalls
    metrics['f1'] = f1
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['PPV'] = pred_value(cm)
    metrics['NPV'] = pred_value(cm, positive=False)

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
            print("Using Contiguity Threshold {} for Clip {}".format(cfg['CLIP_PREDICTION']['CONTIGUITY_THRESHOLD'], str(i)))
            contiguity_threshold = cfg['CLIP_PREDICTION']['CONTIGUITY_THRESHOLD']
            clip_pred_class = max_contiguous_pleural_preds(pred_probs, class_thresh, contiguity_threshold)
        elif clip_pred_method == 'max_sliding_window':
            window_size = cfg['CLIP_PREDICTION']['WINDOW_SIZE']
            clip_pred_class = max_sliding_window(pred_probs, window_size)
        elif clip_pred_method == 'longest_window':
            window_certainty = cfg['CLIP_PREDICTION']['WINDOW_CERTAINTY']
            clip_pred_class = longest_window(pred_probs, window_certainty, class_thresh)
        elif clip_pred_method == 'contiguity_threshold_with_smoothing':
            print("Using Contiguity Threshold {} with Window {} for Clip {}".format(cfg['CLIP_PREDICTION']['CONTIGUITY_THRESHOLD'], cfg['CLIP_PREDICTION']['SMOOTHING_WINDOW'], str(i)))
            contiguity_threshold = cfg['CLIP_PREDICTION']['CONTIGUITY_THRESHOLD']
            smoothing_window = cfg['CLIP_PREDICTION']['SMOOTHING_WINDOW']
            clip_pred_class = contiguous_pleural_with_smoothing_preds(pred_probs, class_thresh, contiguity_threshold,smoothing_window)
        else:
            clip_pred_class, clip_pred_prob = avg_clip_prediction(pred_probs, class_thresh)
            all_pred_probs[i] = clip_pred_prob
        clip_pred_classes.append(clip_pred_class)         # Record predicted class for the clips

    if calculate_metrics:
        clip_labels = clips_df['class']
        metrics = compute_metrics(np.array(clip_labels), np.array(clip_pred_classes), all_pred_probs)
        json.dump(metrics, open(os.path.join(cfg['PATHS']['METRICS'], 'clips_' +
                                             datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.json'), 'w'))

    # Save predictions
    pred_df = pd.DataFrame(clip_pred_classes, columns=['Pred Class'])
    if all_pred_probs is not None:
        pred_df['Pred probs'] = all_pred_probs
    pred_df.insert(0, 'filename', clips_df['filename'])
    pred_df.insert(1, 'class', clips_df['class'])
    pred_df.to_csv(os.path.join(cfg['PATHS']['BATCH_PREDS'] + 'clip_predictions' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv'))
    return pred_df

def contiguous_pleural_with_smoothing_preds(pred_probs,class_thresh,contiguity_threshold,smoothing_window):
    '''
    Smoothes a series of saved frame-level predictions using a moving average and predicts the clip-level class as pleural if there is at least 1
    instance of contiguity_thresh contiguous smoothed frames for which the prediction is pleural.
    :param pred_probs: path to frame-level prediction probabilities
    :class_thresh: classification threshold for pleural views
    :contiguity threshold: number of smoothed frames that must be greater than class_thresh to assign the pleural clip-level class
    :smoothing_window: moving average window (number of consecutive frames over which to average)
    '''
    pred_probs_df = pd.DataFrame(pred_probs,columns=['Pleural View Probability'])
    pred_probs_df['Moving Average Probability'] = pred_probs_df['Pleural View Probability'].rolling(window=smoothing_window).mean()
    pred_probs_df.dropna(inplace=True) # drop null moving average entries
    moving_average_probs = pred_probs_df['Moving Average Probability'].values

    return contiguous_pleural_preds(moving_average_probs,class_thresh,contiguity_threshold)

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
    :param window_size int: length of sliding window
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


def contiguous_pleural_preds(pred_probs, class_thresh, contiguity_thresh=19):#contiguity_thresh):
    '''
    Predicts a clip-level class as pleural if there is at least 1 instance of contiguity_thresh contiguous frames for
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

def longest_window(pred_probs, window_certainty, class_thresh):
    '''
    Predicts a clipwise class by determining which class has the least difference between the class score and any
    average framewise probability taken over a sliding framewise window.
    :param pred_probs (np.array) [n_frames]: Framewise prediction probabilities
    :param window_certainty float: threshold for prediction certainty to being included in the window
    :return (int): Clip class prediction
    '''
    #a window is a pair of a class and window length
    long_window = [-1, 0]
    curr_window = [-1, 0]
    for i in range(pred_probs.shape[0]):
        # obtain class using window certainty threshold
        pred_class = -1
        if pred_probs[i] < window_certainty:
            pred_class = 0
        elif pred_probs[i] > 1-window_certainty:
            pred_class = 1
        else:
            curr_window = [-1, 0]
            continue
        # either increment the current window or start a new window based on this frame's class
        if curr_window[0] == pred_class:
            curr_window[1] += 1
        else:
            curr_window = [pred_class, 1]
        # check if longest window needs to be updated
        if curr_window[1] > long_window[1]:
            long_window = curr_window
    # return the class of the longest window
    if long_window[0] == -1:
        return avg_clip_prediction(pred_probs, class_thresh)[0]
    return long_window[0]


def compute_frame_predictions(cfg, dataset_files_path, class_thresh=0.5, calculate_metrics=True,fold=None):
    '''
    For a particular dataset, make predictions for each image and compute metrics. Save the resultant metrics.
    :param cfg: project config
    :param dataset_files_path: Path to CSV of Dataframe linking filenames to labels
    :param class_thresh: Classification threshold for frame prediction
    :param calculate_metrics: If True, calculate metrics for these predictions; if so, ensure you have a ground truth column
    :param fold: If not None, the cross-validation fold (0-9) being predicted on
    '''
    model_type = cfg['TRAIN']['MODEL_DEF']
    _, preprocessing_fn = get_model(model_type)
    if fold is not None:
        model = load_model(cfg['PATHS']['MODEL_TO_LOAD'] + 'fold{}.h5'.format(fold), compile=False)
    else:
        model = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)
    set_name = dataset_files_path.split('/')[-1].split('.')[0] + '_frames'
    files_df = pd.read_csv(dataset_files_path)

    # Make predictions for each image
    pred_classes, pred_probs = predict_set(model, preprocessing_fn, files_df, threshold=class_thresh,fold=fold)

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
    if fold is not None:
        pred_probs_df.to_csv(os.path.join(cfg['PATHS']['BATCH_PREDS'], 'frame_predictions' +
                                          datetime.datetime.now().strftime('%Y%m%d') + '_fold' + str(fold) + '.csv'))
    else:
        pred_probs_df.to_csv(os.path.join(cfg['PATHS']['BATCH_PREDS'], 'frame_predictions' +
                                      datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv'))
    return pred_probs_df

def pleural_clip_prediction_parameter_experiment(frame_preds_path,min_thresholds,max_thresholds,fold=None,class_thresh_inc=0.1):
    '''
    Performs a grid search for the clip prediction hyperparameters and saves the metrics corresponding to each hyperparameter set considered.

    :param frame_preds_path: path to the frame-wise predictions
    :param min_thresholds: list of minimum thresholds to be considered [contiguity, classification, MA window].
    :param max_thresholds: list of maximum thresholds to be considered [contiguity, classification, MA window].
    To exclude a parameter, set it's value to be the same as in min_thresholds.
    :param fold: If not None, cross-validation fold (0-9) being predicted on.
    :class_thresh_inc: Increment between each classification threshold included in search
    '''
    # Set ranges for each hyperparameter
    contiguity_range = range(min_thresholds[0], max_thresholds[0] + 1)
    classification_range = list(np.arange(min_thresholds[1], max_thresholds[1]+class_thresh_inc, class_thresh_inc))
    window_range = range(min_thresholds[2], max_thresholds[2] + 1)

    # Load frame-level predictions
    preds_df1 = pd.read_csv(frame_preds_path)

    # Extract clip id from frame path
    preds_df1['Clip'] = preds_df1.apply(lambda row: str(row['Frame Path'][:row['Frame Path'].rfind('_')]), axis=1)

    # Create a dataframe to store all metrics
    metrics_df = pd.DataFrame()

    # Loop through all moving average windows
    for window in window_range:

        preds_df = preds_df1.copy()

        # Compute moving average for all clips
        for clip in list(preds_df['Clip'].unique()):
            clip_files_df = preds_df.loc[preds_df['Clip']==clip].copy()
            ind = clip_files_df.index.tolist()
            clip_files_df['Moving Average Probability'] = clip_files_df['Pleural View Probability'].rolling(window=window).mean()
            preds_df.loc[ind[0]:ind[-1],'Moving Average Probability'] = clip_files_df[['Moving Average Probability']]
        preds_df.dropna(inplace=True) # Drop rows with empty moving average predictions

        # Loop through all classification thresholds
        for class_thresh in classification_range:
            # Compute maximum number of contiguous frames meeting class_thresh for each clip
            n_pleural_col = 'Contiguous Predicted Pleural Views'
            preds_df[n_pleural_col] = preds_df['Moving Average Probability'].ge(class_thresh).astype(int)
            contiguity_fnc = lambda x: max_contiguous_pleural_preds_from_series(x, class_thresh=class_thresh)
            clips_df = preds_df.groupby('Clip').agg({'Class': 'max', n_pleural_col: contiguity_fnc})
            # Loop through all contiguity thresholds
            for threshold in contiguity_range:
                # Set prediction as pleural if maximum number of contiguous frames exceeds contiguity threshold
                clips_df['Pred Class'] = clips_df[n_pleural_col].ge(threshold).astype(int)
                # Compute metrics and store in metrics_df
                metrics = compute_metrics(np.array(clips_df['Class']),np.array(clips_df['Pred Class']))
                metrics_flattened = pd.json_normalize(metrics,sep='_')
                if fold is not None:
                    threshold_df = pd.DataFrame(data=[[fold, threshold,class_thresh,window]],columns=['fold','Contiguity Threshold', 'Classification Threshold', 'Moving Average Window'])
                else:
                    threshold_df = pd.DataFrame(data=[[threshold,class_thresh,window]],columns=['Contiguity Threshold', 'Classification Threshold', 'Moving Average Window'])
                new_metrics = threshold_df.merge(metrics_flattened,left_index=True, right_index=True)
                metrics_df = pd.concat([metrics_df, new_metrics],axis=0)
    # Save grid-search metrics
    if fold is not None:
        metrics_df.to_csv(cfg['PATHS']['EXPERIMENTS'] + 'pleural_contiguity_thresholds_fold{}_new2.csv'.format(fold),index=False)
    else:
        metrics_df.to_csv(cfg['PATHS']['EXPERIMENTS'] + 'pleural_contiguity_thresholds_holdout.csv',index=False)
    return metrics_df


def max_contiguous_pleural_preds_from_series(pred_series,class_thresh=0.5):
    '''
    Computes the maximum number of contiguous frames exceeding a given classification threshold froma  series of frame-level predictions.
    :param pred_series: series of frame-level prediction probabilities
    :class_thresh: classification threshold for pleural views
    :return max_contiguous: the maximu number of contiguous frames that were considered pleural (above class_thresh)
    '''
    pred_probs = np.asarray(pred_series)
    max_contiguous = cur_contiguous = 0
    for i in range(pred_probs.shape[0]):
        if pred_probs[i] >= class_thresh:
            cur_contiguous += 1
        else:
            cur_contiguous = 0
        if cur_contiguous > max_contiguous:
            max_contiguous = cur_contiguous
    return max_contiguous


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

def pred_value(cm,positive=True):
    '''
    Computes positive or negative predictive value given a confusion matrix.
    :param cm: confusion matrix
    :param positive: whether to compute the predictive value of the positive (True) or negative (False) class
    :return PV: predictive value
    '''
    TN = cm[0][0]
    FP = cm[0][1]
    TP = cm[1][1]
    FN = cm[1][0]
    if positive:
        PV = TP/(TP + FP)
    else:
        PV = TN/(TN + FN)
    return PV



if __name__ == '__main__':
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

    # -- Compute clip-predictions
    # frames_path = cfg['PATHS']['FRAMES_TABLE']
    # clips_path = cfg['PATHS']['CLIPS_TABLE']
    # compute_clip_predictions(cfg, frames_path, clips_path, clip_pred_method=cfg['CLIP_PREDICTION']['CLIP_PREDICTION_METHOD'],
    #                          class_thresh=cfg['CLIP_PREDICTION']['CLASSIFICATION_THRESHOLD'], calculate_metrics=False, for_sprints=True)

    # -- Perform grid-search for clip-prediction hyperparameters
    for fold in range(10):
        print('Fold: {}'.format(fold))
        frame_preds_path = cfg['PATHS']['BATCH_PREDS'] + 'frame_predictions20220701_fold{}.csv'.format(fold)
        pleural_clip_prediction_parameter_experiment(frame_preds_path,
                                                     [1, 0.6, 1],
                                                     [30, 0.8, 30],
                                                     fold=fold,
                                                     class_thresh_inc=0.01)