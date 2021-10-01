import datetime
import os
import io

import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np
import yaml

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

def plot_roc(labels, predictions, class_name_list, dir_path=None, title=None):
    '''
    Plots the ROC curve for predictions on a dataset
    :param labels: Ground truth labels
    :param predictions: Model predictions corresponding to the labels
    :param class_name_list: Ordered list of class names
    :param dir_path: Directory in which to save image
    '''
    plt.clf()
    for class_id in range(len(class_name_list)):
        class_name = class_name_list[class_id]
        single_class_preds = predictions[:, class_id]    # Only care about one class
        single_class_labels = (np.array(labels) == class_id) * 1.0
        fp, tp, _ = roc_curve(single_class_labels, single_class_preds)  # Get values for true positive and true negative
        plt.plot(100*fp, 100*tp, label=class_name, linewidth=2)   # Plot the ROC curve

    if title is None:
        plt.title('ROC curves for test set')
    else:
        plt.title(title)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-5,105])
    plt.ylim([-5,105])
    plt.grid(True)
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal')
    if dir_path is not None:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(dir_path + 'ROC_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return plt

def plot_confusion_matrix(labels, predictions, class_name_list, dir_path=None, title=None):
    '''
    Plot a confusion matrix for the ground truth labels and corresponding model predictions for a particular class.
    :param labels: Ground truth labels
    :param predictions: Model predictions
    :param class_name_list: Ordered list of class names
    :param dir_path: Directory in which to save image
    '''
    plt.clf()
    predictions = list(np.argmax(predictions, axis=1))
    ax = plt.subplot()
    cm = confusion_matrix(list(labels), predictions)  # Determine confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # Plot confusion matrix
    ax.figure.colorbar(im, ax=ax)
    ax.set(yticklabels=class_name_list, xticklabels=class_name_list)
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=1, offset=0.5))
    ax.yaxis.set_major_locator(mpl.ticker.IndexLocator(base=1, offset=0.5))

    # Print the confusion matrix numbers in the center of each cell of the plot
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    # Set plot's title and axis names
    if title is None:
        plt.title('Confusion matrix for test set')
    else:
        plt.title(title)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # Save the image
    if dir_path is not None:
        plt.savefig(dir_path + 'CM_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')

    print('Confusion matrix: ', cm)    # Print the confusion matrix
    return plt