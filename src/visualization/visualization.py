import datetime
import os
import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import yaml
from pandas.api.types import is_numeric_dtype
from skopt.plots import plot_objective
import matplotlib.patches as patches
import math

mpl.rcParams['figure.figsize'] = (12, 8)
cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

def plot_to_tensor():
    '''
    Converts a matplotlib figure to an image tensor
    :param figure: A matplotlib figure
    :return: Tensorflow tensor representing the matplotlib image
    '''
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image_tensor = tf.image.decode_png(buf.getvalue(), channels=4)     # Convert .png buffer to tensorflow image
    image_tensor = tf.expand_dims(image_tensor, 0)     # Add the batch dimension
    return image_tensor


def get_roc_data(path_to_preds, base_fpr=np.linspace(0, 1, 1001)):
    '''
    Computes the true positive rate for plotting an ROC curve and the curves corresponding AUC.
    :param path_to_preds: path to frame-level prediction probabilities.
    :param base_fpr: fpr-vector that will be used for plotting.
    '''
    frame_df = pd.read_csv(path_to_preds)
    b_fpr, b_tpr, _ = roc_curve(frame_df['Class'], frame_df['Pleural View Probability'])
    b_tpr_itpls = np.interp(base_fpr, b_fpr, b_tpr)  # Get tpr for plotting
    aucs = auc(base_fpr, b_tpr_itpls)  # Compute AUC

    return b_tpr_itpls, aucs


def get_roc_data_kfold(paths_to_fold_preds, base_fpr=np.linspace(0, 1, 1001)):
    '''
    Computes the necessary information required for plotting the average ROC curve for a kfold cross validation experiment.
    :paths_to_fold_preds: list of paths to the prediction probabilities for each fold
    :param base_fpr: fpr-vector that will be used for plotting.
    :return b_mean_tprs: average true positive rate across all k folds
    :return b_tprs_lower: lower bound for the range of ROC curves to be plotted (average - std)
    :return b_tprs_upper: upper bound for the range of ROC curves to be plotted (average + std)
    :return avg_b_auc: mean AUC across all k folds
    :return std_b_auc: standard deviation of the AUC across all k folds
    '''

    auck, b_tpr_itplsk = [], []
    cms = []

    # Compute true positive rate and AUC for each fols
    for path in paths_to_fold_preds:
        b_tpr_itpls, aucs = get_roc_data(path, base_fpr=base_fpr)
        b_tpr_itplsk.append(b_tpr_itpls)
        auck.append(aucs)

    b_tpr_itplsk = np.array(b_tpr_itplsk)
    b_mean_tprs = b_tpr_itplsk.mean(axis=0)  # Mean tpr across all folds
    b_std = b_tpr_itplsk.std(axis=0)  # Standard deviation of the tpr across all folds
    b_tprs_upper = np.minimum(b_mean_tprs + b_std, 1)  # Lower bound for plotting
    b_tprs_lower = b_mean_tprs - b_std  # Upper bound for plotting

    avg_b_auc = np.mean(np.array(auck))  # Mean AUC across all folds
    std_b_auc = np.std(np.array(auck))  # Standard deviation of the AUC across all folds

    return b_mean_tprs, b_tprs_lower, b_tprs_upper, avg_b_auc, std_b_auc


def plot_roc(path_to_preds, kfold=False, base_fpr=np.linspace(0, 1, 1001), c='g', tit=None, fig=None, ax=None,
             save_name=None):
    '''
    Plots the ROC curve for predictions on a dataset.
    :param path_to_preds: Path to frame-level prediction probabilities. If kfold=True, this contains a list of paths to each fold's predictions.
    :param kfold: If True, the average ROC curve across all folds will be plotted.
    :param base_fpr: fpr-vector for plotting
    :param c: Colour to plot the ROC curve with
    :param tit: If not None, the title of plot
    :param fig: If ROC curve will be a subplot, set to fig object of entire plot.
    :param ax: If ROC curve will be a subplot, set this equal to its corresponding axis.
    :param save_name: If not None, the ROC curve will be saved under this name.
    '''
    if kfold:
        b_tpr_itpls, b_tprs_lower, b_tprs_upper, avg_b_auc, std_b_auc = get_roc_data_kfold(path_to_preds)
    else:
        b_tpr_itpls, aucs = get_roc_data(path_to_preds)

    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.plot(base_fpr, b_tpr_itpls, c)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='dashed', linewidth=2)
    if kfold:
        ax.fill_between(base_fpr, b_tprs_lower, b_tprs_upper, color=c, alpha=0.2)
        ax.legend(["ROC (AUC = {:.3f} \u00B1 {:.3f})".format(avg_b_auc, std_b_auc), "Naive classifier"],
                  loc='lower right', fontsize=14)
    else:
        ax.legend(["ROC (AUC = {:.3f})".format(aucs), "Naive classifier"], loc='lower right', fontsize=14)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xticks(np.arange(0.0, 1.1, 0.1), minor=False)
    ax.set_xticks(np.arange(0.0, 1.05, 0.05), minor=True)
    ax.set_yticks(np.arange(0.0, 1.1, 0.1), minor=False)
    ax.set_yticks(np.arange(0.0, 1.05, 0.05), minor=True)
    ax.set_ylabel('True Positive Rate', fontsize=16)
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_autoscale_on(False)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    if tit is not None:
        ax.set_title(tit, fontweight='bold', size=16, loc='left')

    if save_name is not None:
        plt.savefig(cfg['PATHS']['IMAGES'] + save_name + '_ROC' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')


def get_confusion_matrix_data(path_to_preds, frames=False, get_labels=True):
    '''
    Computes the confusion matrix for ground truth labels and corresponding model predictions fo a given dataset.
    :path to preds: path to clip or frame-level predictions
    :frames: If True, the predictions are at the frame-level (i.e. probabilities)
    :get_labels: If True, an array of value (percent) labels are also returned, to be used for plotting
    '''
    df = pd.read_csv(path_to_preds) # Load predictions

    if frames:     # Compute frame-level class prediction
        df['Pred Class'] = df.apply(lambda row: 1 if row['Pleural View Probability'] >= 0.5 else 0, axis=1)

    cm = np.vstack(confusion_matrix(df['Class'], df['Pred Class']).tolist())     # Compute confusion matrix

    if get_labels: # Create label array for plotting
        vals = ["{:,.0f}".format(f) for f in cm.flatten()]
        percents = ["{0:.1%}".format(f) for f in cm.flatten() / np.sum(cm)]
        labels = [f"{m}\n\n{p}" for m, p, in zip(vals, percents)]
        labels = np.asarray(labels).reshape(2, 2)
        return cm, labels
    else:
        return cm


def get_confusion_matrix_data_kfold(paths_to_fold_preds, frames=False):
    '''
    Computes the average confusion matrix for a kfold cross validation experiment.
    :paths_to_fold_preds: list of paths to clip or frame-level predictions for each fold
    :frames: If True, the predictions are at the frame-level (i.e. probabilities)
    :return mean_cm: Average confusion matrix across all folds
    :return std_cm: Standard deviation of the confusion matrix across all folds
    :return labels: Array of mean +/- std labels for plotting
    '''

    all_cm = []
    for path in paths_to_fold_preds: # Compute confusion matrices for each fold, storing them in all_cm
        all_cm.append(np.vstack(get_confusion_matrix_data(path, frames=frames, get_labels=False)))
    mean_cm = np.mean(all_cm, axis=0).astype(int) # Mean confusion matrix across folds
    std_cm = np.std(all_cm, axis=0).astype(int) # Standard deviation confusion matrix across folds

    # Get labels for plotting
    group_means = ["{:,.0f}".format(f) for f in mean_cm.flatten()]
    group_stds = ["{:,.0f}".format(s) for s in std_cm.flatten()]
    group_mean_percent = ["{0:.1%}".format(f) for f in mean_cm.flatten() / np.sum(mean_cm)]
    group_std_percent = ["{0:.1%}".format(s) for s in std_cm.flatten() / np.sum(mean_cm)]
    labels = [f"{m} \u00B1 {s}\n\n{mp} \u00B1 {sp}" for m, s, mp, sp in
              zip(group_means, group_stds, group_mean_percent, group_std_percent)]
    labels = np.asarray(labels).reshape(2, 2)

    return mean_cm, std_cm, labels


def plot_confusion_matrix(path_to_preds, frames=False, kfold=False, categories=['Parenchymal', 'Pleural'], c="Blues",
                          tit=None, fig=None, ax=None, save_name=None):
    '''
    Plot a confusion matrix for the ground truth labels and corresponding model predictions.
    :param path to preds: Path to clip or frame-level predictions. If kfold=True, this contains a list of paths to each fold's predictions.
    :param frames: If True, the predictions are at the frame-level (i.e. probabilities)
    :param kfold: If True, the average confusion matrix across all folds will be plotted.
    :param categories: Ordered list of class names (ith element corresponds to the ith class)
    :param c: Colormap for plotting
    :param tit: If not None, title of plot
    :param fig: If CM will be a subplot, set to fig object of entire plot.
    :param ax: If CM will be a subplot, set this equal to its corresponding axis.
    :param save_name: If not None, the confusion matrix will be saved under this name.
    '''

    if kfold:
        cm, cm_std, labels = get_confusion_matrix_data_kfold(path_to_preds, frames=frames)
    else:
        cm, labels = get_confusion_matrix_data(path_to_preds, frames=frames)

    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    cm_plot = sns.heatmap(cm, annot=labels, fmt="", cmap=c, xticklabels=categories, yticklabels=categories,
                          annot_kws={"size": 14}, ax=ax)
    cm_plot.set_xticklabels(cm_plot.get_xmajorticklabels(), fontsize=12)
    cm_plot.set_yticklabels(cm_plot.get_ymajorticklabels(), fontsize=12, va='center')
    cbar = cm_plot.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cm_plot.set_ylabel('True class', fontsize=16)
    if frames:
        cm_plot.set_xlabel('Predicted (frame-level) class', fontsize=16)
    else:
        cm_plot.set_xlabel('Predicted (clip-level) class', fontsize=16)
    if tit is not None:
        cm_plot.set_title(tit, fontweight='bold', size=16, loc='left')

    if save_name is not None:
        plt.savefig(cfg['PATHS']['IMAGES'] + save_name + '_CM' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')

    return cm_plot, cm


def plot_fig5(hd_clip_preds, hd_frame_preds, kfold_clip_preds, kfold_frame_preds, save_name=None):
    '''
    Plots Figure 5 from the manuscript.
    :param hd_clip_preds: path to clip-level predictions on the holdout set
    :param hd_frame_preds: path to frame-level predictions on the holdout set
    :param kfold_clip_preds: list of paths to clip-level predictions on each fold of the validation set
    :param kfold_frame_preds: list of paths to frame-level predictions on each fold of the validation set
    :param save_name: If not None, the Figure will be saved under this name.
    '''
    fig5, axs = plt.subplots(3, 2, figsize=(14, 16))

    # kfold roc
    k_roc = plot_roc(kfold_frame_preds, fig=fig5, ax=axs[0, 0], kfold=True, tit="(A)")

    # holdout roc
    hd_roc = plot_roc(hd_frame_preds, fig=fig5, ax=axs[0, 1], tit="(B)")

    # kfold frame cm
    k_fr = plot_confusion_matrix(kfold_frame_preds, frames=True, kfold=True, c="Greens", fig=fig5, ax=axs[1, 0],
                                 tit="(C)")

    # holdout frame cm
    hd_fr = plot_confusion_matrix(hd_frame_preds, frames=True, c="Greens", fig=fig5, ax=axs[1, 1], tit="(D)")

    # kfold clip cm
    k_cl = plot_confusion_matrix(kfold_clip_preds, kfold=True, c="Greens", fig=fig5, ax=axs[2, 0], tit="(E)")

    # holdout clip cm
    hd_cl = plot_confusion_matrix(hd_clip_preds, fig=fig5, ax=axs[2, 1], tit="(F)")

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    if save_name is not None:
        plt.savefig(cfg['PATHS']['IMAGES'] + save_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')

    return fig5


def plot_pleural_probability(frame_df, filename, patch_frame=None, window=17, contiguity_thresh=7,
                             classification_thresh=0.7, fig=None, ax=None, tit=None, save_name=None):
    '''
    Plots the raw and smoothed predicted pleural probabilities as a function of the frame number for a given clip.
    :param frame_df: dataframe containing frame-level prediction probabilities for at least one clip
    :param filename: filename of the clip to plot
    :patch_frame: If not None, a patch of width window will be drawn starting from this frame number.
    If it's value exceeds the maximum allowable frame number (total frames - window), the patch will be applied at the end of the clip by default.
    :param window: Width of the moving average window used to smooth the frame-level predictions.
    :param contiguity_thresh: Contiguity threshold used in the clip-prediction algorithm. Will specify the x-tick interval.
    :param classification_thresh: Classification threshold used in the clip-prediction algorithm.
    :param fig: If not None, the fig object that the plot will be plotted in.
    :param ax: If not None, the axis object that the plot will be plotted along.
    :tit: If not None, is the title of the plot
    :param save_name: If not None, the figure will be saved under this name.
    '''

    if 'filename' not in list(frame_df.columns):  # Get clip filename from frame path if not already done
        frame_df['filename'] = frame_df.apply(lambda row: row['Frame Path'][:row['Frame Path'].rfind('_')], axis=1)

    if ax is None or fig is None:  # Initialize figure if not already done
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot raw prediction probabilities
    clip_files_df = frame_df.loc[frame_df['filename'] == filename]
    x_raw = range(len(clip_files_df))
    y_raw = clip_files_df['Pleural View Probability'].reset_index(drop=True)

    # Get smoothed prediction probabilities
    smoothed_probs = clip_files_df['Pleural View Probability'].rolling(window=window).mean().reset_index(drop=True)
    y_smooth = smoothed_probs[window:]
    x_smooth = np.arange(window / 2, window / 2 + len(y_smooth), 1)

    # Plot probabilities
    ax.plot(x_raw, y_raw, linewidth=1)
    ax.plot(x_smooth, y_smooth, linewidth=2, color='k')

    # Add red patch over a desired window of frames
    if patch_frame is not None:
        if patch_frame > len(clip_files_df)-window-1:  # if patch_frame exceeds allowable frames, default to end of clip
            patch_frame = len(clip_files_df)-window-1
        w0 = patch_frame
        wind = y_raw[w0:w0 + window]
        wm = min(wind)
        wM = max(wind)
        rect = patches.Rectangle((w0, 0), window, 1, linewidth=1, facecolor='k', alpha=0.1)
        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.scatter(window / 2 + w0, smoothed_probs[window + w0], s=50, c='r', zorder=3)
        ax.plot(x_raw[w0:w0 + window + 1], y_raw[w0:w0 + window + 1], color='r', linewidth=1)  # range(w0,w0+17)

    # Format plot
    ax.set_xlabel('Frame Number', fontsize=16)
    ax.set_ylabel('Pleural Prediction Probability', fontsize=16)
    ax.axhline(y=classification_thresh, linestyle='-.', linewidth=1, color='dimgray')  # Plot classification threshold
    ax.set_xticks(
        np.arange(min(x_raw), max(x_raw) + 1, contiguity_thresh))  # Get xticks at contiguity threshold intervals
    ax.grid(axis='x', linestyle='--', linewidth=1)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, len(clip_files_df) - 1])
    if tit is not None:
        ax.set_title(tit, size=16, loc='left')

    # Save
    if save_name:
        plt.savefig(cfg['PATHS']['IMAGES'] + save_name + '_prob-v-time' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')

    return plt


def plot_multiple_probability_time_plots(frame_df, clips, tick_mod=None, titles=None, save_name=None, figsize=(14,10)):
    '''
    Plots multiple probability-time plots in one Figure as multiple subplots. See Figure 7 and A2 from the manuscript for an example.
    :param frame_df: Dataframe containing pleural prediction probabilities of all clips
    :param clips: Ordered list of clip filenames to include (tp, tn, fp, fn)
    :param tick_mod: If not None, list of integers, specifying how many x-ticks to label for each subplot. Defaults to every x-tick getting labelled.
    :param titles: If not None, list of titles for each subplot
    :param save_name: If not None, the figure will be saved under this name.
    :param figsize: Size of the figure.
    '''

    frame_df['filename'] = frame_df.apply(lambda row: row['Frame Path'][:row['Frame Path'].rfind('_')],
                                          axis=1)  # Get clip filename from frame path

    fig, axs = plt.subplots(math.ceil(len(clips)/2), 2, figsize=figsize)  # Initialize plot

    for i, filename in enumerate(clips):  # Plot curve for each clip using plot_pleural_probability

        ax = axs.flat[i]
        tit = None
        if titles is not None:
            tit = titles[i]

        plot_pleural_probability(frame_df, filename, ax=ax, fig=fig, tit=tit)

        count = 0
        for label in ax.xaxis.get_ticklabels():  # Only label every tick_mod[i] x-ticks
            if count % tick_mod[i] != 0:
                label.set_visible(False)
            count += 1

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.3,
                        hspace=0.3)

    if save_name:
        plt.savefig(cfg['PATHS']['IMAGES'] + save_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')

    return fig


def visualize_heatmap(orig_img, heatmap, img_filename, label, prob, class_names, dir_path=None):
    '''
    Obtain a comparison of an original image and heatmap produced by Grad-CAM.
    :param orig_img: Original X-Ray image
    :param heatmap: Heatmap generated by Grad-CAM.
    :param img_filename: Filename of the image explained
    :param label: Ground truth class of the example
    :param probs: Prediction probabilities
    :param class_names: Ordered list of class names
    :param dir_path: Path to save the generated image
    :return: Path to saved image
    '''

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(orig_img)
    ax[1].imshow(heatmap)

    # Display some information about the example
    pred_class = np.round(prob)
    fig.text(0.02, 0.90, "Prediction probability: " + str(prob), fontsize=10)
    fig.text(0.02, 0.92, "Predicted Class: " + str(int(pred_class)) + ' (' + class_names[int(pred_class)] + ')', fontsize=10)
    if label is not None:
        fig.text(0.02, 0.94, "Ground Truth Class: " + str(label) + ' (' + class_names[label] + ')', fontsize=10)
    fig.suptitle("Grad-CAM heatmap for image " + img_filename, fontsize=8, fontweight='bold')
    fig.tight_layout()

    # Save the image
    filename = None
    if dir_path is not None:
        filename = os.path.join(dir_path, img_filename.split('/')[-1] + '_gradcam_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
        plt.savefig(filename)
    return filename


def plot_clip_pred_threshold_experiment_old(metrics_df, var_col, metrics_to_plot=None,
                                        ax=None, im_path=None, title=None, x_label=None):
    '''
    Visualizes the Plot classification metrics for clip predictions over various pleural view count thresholds.
    :param metrics_df: DataFrame containing classification metrics for different. The first column should be the
                       various contiguity thresholds and the rest are classification metrics
    :min_threshold: Minimum contiguity threshold
    :max_threshold: Maximum contiguity threshold
    :thresh_col: Column of DataFrame corresponding to threshold variable
    :class_thresh: Classification threshold
    :metrics_to_plot: List of metrics to include on the plot
    :ax: Matplotlib subplot
    :im_path: Path in which to save image
    :title: Plot title
    :x_label: X-label for plot
    '''
    min_threshold = metrics_df[var_col][0]
    max_threshold = metrics_df[var_col][len(metrics_df[var_col]) - 1]

    min_y_lim = 1.0
    max_y_lim = 0.0
    for x in metrics_to_plot:
        min = metrics_df[x].min()
        max = metrics_df[x].max()
        min_y_lim = min if min < min_y_lim else min_y_lim
        max_y_lim = max if max < max_y_lim else max_y_lim

    if ax is None:
        ax = plt.subplot()
    if title:
        plt.title(title)
    if x_label:
        ax.set_xlabel(var_col)

    if metrics_to_plot is None:
        metric_names = [m for m in metrics_df.columns if m != var_col and is_numeric_dtype(metrics_df[m])]
    else:
        metric_names = metrics_to_plot

    # Plot each metric as a separate series and place a legend
    for metric_name in metric_names:
        if is_numeric_dtype(metrics_df[metric_name]):
            ax.plot(metrics_df[var_col], metrics_df[metric_name])

    # Change axis ticks and add grid
    #ax.minorticks_on()
    # for tick in ax.get_xticklabels():
    #     tick.set_color('gray')
    # for tick in ax.get_yticklabels():
    #     tick.set_color('gray')
    ax.set_xlim(min_threshold - 1, max_threshold + 1)
    ax.set_ylim(min_y_lim-0.02, max_y_lim+0.02)
    ax.xaxis.set_ticks(np.arange(0, max_threshold + 1, 5))
    ax.yaxis.set_ticks(np.arange(min_y_lim-0.02, max_y_lim+0.02, 0.1))
    # ax.grid(True, which='both', color='lightgrey')

    # Draw legend
    ax.legend(metric_names, loc='lower right')
    plt.show()
    if im_path:
        plt.savefig(im_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return ax

def plot_clip_pred_experiment(metrics_df, var_col, metrics_to_plot=None,
                                        im_path=None, title=None, x_label=None,  y_label=None,
                                        model_name = None, experiment_type = None):
    '''
    Visualizes the Plot classification metrics for clip predictions over various contiguity thresholds.
    :param metrics_df: DataFrame containing classification metrics for different. The first column should be the
                       various contiguity thresholds and the rest are classification metrics
    :var_col: Column of DataFrame corresponding to variable
    :metrics_to_plot: List of metrics to include on the plot
    :im_path: Path in which to save image
    :title: Plot title
    :x_label: X-label for plot
    :y_label: X-label for plot
    :model_name: Name of model used to generate predictions being plotted
    :experiment_type: Name of experiment used to generate values being plotted
    '''

    if metrics_to_plot is None:
        metric_names = [m for m in metrics_df.columns if m != var_col and is_numeric_dtype(metrics_df[m])]
    else:
        metric_names = metrics_to_plot

    # Plot each metric as a separate series and place a legend after clearing the plot
    plt.clf()
    for metric_name in metric_names:
        if is_numeric_dtype(metrics_df[metric_name]):
            sns.lineplot(x=metrics_df[var_col], y=metrics_df[metric_name])

    # Draw legend
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(var_col)
    if y_label:
        plt.ylabel(var_col)
    plt.legend(metric_names)
    if im_path:
        savefig_name = im_path
        if model_name:
            savefig_name  = savefig_name + model_name+"-"
        if experiment_type:
            savefig_name = savefig_name + experiment_type+"-"
        plt.savefig(savefig_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return

def plot_bayesian_hparam_opt(model_name, hparam_names, search_results, save_fig=False):
    '''
    Plot all 2D hyperparameter comparisons from the logs of a Bayesian hyperparameter optimization.
    :param model_name: Name of the model
    :param hparam_names: List of hyperparameter identifiers
    :param search_results: The object resulting from a Bayesian hyperparameter optimization with the skopt package
    :param save_fig:
    :return:
    '''

    # Abbreviate hyperparameters to improve plot readability
    axis_labels = hparam_names.copy()
    for i in range(len(axis_labels)):
        if len(axis_labels[i]) >= 12:
            axis_labels[i] = axis_labels[i][:4] + '...' + axis_labels[i][-4:]

    # Plot
    axes = plot_objective(result=search_results, dimensions=axis_labels)

    # Create a title
    fig = plt.gcf()
    fig.suptitle('Bayesian Hyperparameter\n Optimization for ' + model_name, fontsize=15, x=0.65, y=0.97)

    # Indicate which hyperparameter abbreviations correspond with which hyperparameter
    hparam_abbrs_text = ''
    for i in range(len(hparam_names)):
        hparam_abbrs_text += axis_labels[i] + ':\n'
    fig.text(0.50, 0.8, hparam_abbrs_text, fontsize=10, style='italic', color='mediumblue')
    hparam_names_text = ''
    for i in range(len(hparam_names)):
        hparam_names_text += hparam_names[i] + '\n'
    fig.text(0.65, 0.8, hparam_names_text, fontsize=10, color='darkblue')

    fig.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(cfg['PATHS']['EXPERIMENT_VISUALIZATIONS'], 'Bayesian_opt_' + model_name + '_' +
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png'))