import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.metrics as m
from sklearn.metrics import roc_curve, auc

from itertools import chain
from . import utils


def get_output_df_2D(dataset, model, label_mode, thresh=0.5):

    preds, labels, probs = [], [], []

    # Iterate through the dataset and nucleus IDs
    for x, ys in dataset:
        new_probs = model.predict(x, verbose=0)
        
        if label_mode == "binary":
            preds.append((new_probs >= thresh).astype(float))
            probs.append(new_probs)
            labels.append(ys.numpy())
        elif label_mode == "categorical":
            preds.append(np.argmax(new_probs, axis=1))
            probs.append(new_probs[:, 1])
            labels.append(np.argmax(ys.numpy(), axis=1))

    # Create a Pandas DataFrame from the collected data
    return pd.DataFrame(
        {
            "im_path": dataset.file_paths,
            "label": [int(e[0]) for e in list(chain(*labels))],
            "prob": [e[0] for e in list(chain(*probs))],
            "pred": [int(e[0]) for e in list(chain(*preds))],
            "Z_pos": utils.get_nuc_Z_position(dataset.file_paths)
        }
    )



def get_history_df(hist):
    """
    Convert training history dictionary to a tidy DataFrame.

    This function takes a training history dictionary typically obtained from
    training a machine learning model and converts it into a tidy DataFrame.
    The resulting DataFrame contains metrics such as accuracy, AUC, and F1-score
    for both training and validation datasets over epochs.

    Args:
        hist (dict): A dictionary containing training history data with metrics.

    Returns:
        pd.DataFrame: A tidy DataFrame with columns 'metric', 'value', 'epoch',
                      and 'dataset' representing metric name, metric value,
                      epoch number, and dataset type (train or val).
    """

    # Create a DataFrame from the history dictionary
    hist_df = pd.DataFrame.from_dict(hist)

    # Define the list of metrics to include in the DataFrame
    metrics = [
        "accuracy",
        "val_accuracy",
        "auc",
        "val_auc",
        "f1_score",
        "val_f1_score",
    ]
    hist_df = hist_df[metrics]

    # Determine the number of rows and columns in the DataFrame
    n_rows, n_cols = hist_df.shape

    # Create a list of epochs (1, 2, ..., n_rows)
    eps = list(range(1, n_rows + 1))

    # Reshape the DataFrame into a tidy format
    hist_df = hist_df.melt(var_name="metric")

    # Add the 'epoch' and 'dataset' columns to the DataFrame
    hist_df["epoch"] = eps * n_cols
    n_metrics = int(n_cols / 2)
    dataset = ["train"] * n_rows + ["val"] * n_rows
    hist_df["dataset"] = dataset * n_metrics

    return hist_df


def plot_training(df):
    """
    Plot training progress using a line plot.

    This function takes a tidy DataFrame containing training progress metrics
    over epochs and generates a line plot to visualize the changes in different
    metrics such as accuracy, AUC, and F1-score for both training and validation
    datasets over epochs.
    """
    sns.lineplot(
        df,
        x="epoch",
        y="value",
        hue="metric",
        style="dataset",
        palette="Paired",
    )
    plt.title("Training Progress")
    plt.show()


def get_output_df_voting(nuc_ids, dataset, model, label_mode, thresh=0.5, cal_model=None):
    """
    Create a DataFrame with voting-based predictions and aggregated probabilities.

    Args:
        nuc_ids (list): A list of nucleus IDs corresponding to the samples.
        dataset (tf.data.Dataset): A TensorFlow Dataset containing samples.
        model (tf.keras.Model): A trained machine learning model for making predictions.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the following columns:
            - 'nuc_id': Nucleus IDs.
            - 'label': True labels for each nucleus.
            - 'mean_prob': Mean of predicted probabilities for each nucleus.
            - 'median_prob': Median of predicted probabilities for each nucleus.
            - 'std_prob': Standard deviation of predicted probabilities for each nucleus.
            - 'hard_preds': Majority voting-based hard predictions for each nucleus.

    This function aggregates predictions and probabilities for each nucleus in the input
    dataset using majority voting. It calculates the mean, median, and standard
    deviation of predicted probabilities and determines hard predictions based on
    majority voting.
    """
    preds_dic = {}  # Store predictions for each nucleus ID
    label_dic = {}  # Store true labels for each nucleus ID
    probs_dic = {}  # Store predicted probabilities for each nucleus ID

    # Iterate through the dataset and nucleus IDs
    for batch_nuc_ids, sample in zip(nuc_ids, dataset):
        x, ys = sample
        new_probs = model.predict(x, verbose=0)
        if cal_model:
            new_probs = cal_model.predict(new_probs)
            
        if label_mode == "binary":
            new_preds = (new_probs >= thresh).astype(float)
            ys = ys.numpy()
        elif label_mode == "categorical":
            new_preds = np.argmax(new_probs, axis=1)
            new_probs = new_probs[:, 1]
            ys = np.argmax(ys.numpy(), axis=1)

        # Process predictions and probabilities for each nucleus in the batch
        for nuc_id, y, pred, prob in zip(
            batch_nuc_ids, ys, new_preds, new_probs
        ):
            if label_mode == "binary":
                y = int(y[0])
            label_dic[nuc_id] = y
            # Add the record to the dictionary, even if key does not exists yet
            preds_dic.setdefault(nuc_id, []).append(pred)
            probs_dic.setdefault(nuc_id, []).append(prob)

    # Create a Pandas DataFrame from the collected data
    return pd.DataFrame(
        {
            "nuc_id": list(label_dic.keys()),
            "label": list(label_dic.values()),
            "mean_prob": [np.mean(p) for p in probs_dic.values()],
            "median_prob": [np.median(p) for p in probs_dic.values()],
            "std_prob": [np.std(p) for p in probs_dic.values()],
            "hard_preds": [
                int(np.median(pred)) for pred in preds_dic.values()
            ],
        }
    )



def predict_df_voting(nuc_ids, dataset, model, label_mode, thresh=0.5, cal_model=None):
    """
    Create a DataFrame with voting-based predictions and aggregated probabilities.

    Args:
        nuc_ids (list): A list of nucleus IDs corresponding to the samples.
        dataset (tf.data.Dataset): A TensorFlow Dataset containing samples.
        model (tf.keras.Model): A trained machine learning model for making predictions.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the following columns:
            - 'nuc_id': Nucleus IDs.
            - 'mean_prob': Mean of predicted probabilities for each nucleus.
            - 'median_prob': Median of predicted probabilities for each nucleus.
            - 'std_prob': Standard deviation of predicted probabilities for each nucleus.
            - 'hard_preds': Majority voting-based hard predictions for each nucleus.

    This function aggregates predictions and probabilities for each nucleus in the input
    dataset using majority voting. It calculates the mean, median, and standard
    deviation of predicted probabilities and determines hard predictions based on
    majority voting.
    """
    preds_dic = {}  # Store predictions for each nucleus ID
    probs_dic = {}  # Store predicted probabilities for each nucleus ID

    # Iterate through the dataset and nucleus IDs
    for batch_nuc_ids, x in zip(nuc_ids, dataset):
        new_probs = model.predict(x, verbose=0)
        if cal_model:
            new_probs = cal_model.predict(new_probs)
            
        if label_mode == "binary":
            new_preds = (new_probs >= thresh).astype(float)
        elif label_mode == "categorical":
            new_preds = np.argmax(new_probs, axis=1)
            new_probs = new_probs[:, 1]

        # Process predictions and probabilities for each nucleus in the batch
        for nuc_id, pred, prob in zip(batch_nuc_ids, new_preds, new_probs):
            # Add the record to the dictionary, even if key does not exists yet
            preds_dic.setdefault(nuc_id, []).append(pred)
            probs_dic.setdefault(nuc_id, []).append(prob)

    # Create a Pandas DataFrame from the collected data
    return pd.DataFrame(
        {
            "nuc_id": list(preds_dic.keys()),
            "mean_prob": [np.mean(p) for p in probs_dic.values()],
            "median_prob": [np.median(p) for p in probs_dic.values()],
            "std_prob": [np.std(p) for p in probs_dic.values()],
            "hard_preds": [
                int(np.median(pred)) for pred in preds_dic.values()
            ],
        }
    )



def get_metrics_df_by_threshold(thresh_df, target_thresh=0.5):
    """
    Filter a DataFrame for a specific threshold and reorganize it for analysis.

    Args:
        thresh_df (pd.DataFrame): A Pandas DataFrame containing threshold-based metrics.
        target_thresh (float): The target threshold value to filter the DataFrame.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing filtered and reorganized metrics.

    This function takes a DataFrame `thresh_df` that typically includes threshold-based
    evaluation metrics and selects rows with a specific threshold value specified by
    `target_thresh`. It then reorganizes the DataFrame for analysis by melting it,
    making it suitable for further visualization or statistical operations.
    """
    # Filter rows with the specified target threshold
    df = thresh_df[np.isclose(thresh_df["threshold"], target_thresh)]

    # Extract 'dataset' values and drop the 'dataset' column
    datasets = df["dataset"].to_list()
    df = df.drop("dataset", axis=1)

    n_columns = len(df.columns)

    # Reshape the DataFrame by melting it
    df = df.melt()

    # Duplicate 'dataset' values to match the reshaped DataFrame
    df["dataset"] = datasets * n_columns

    return df




def plot_conf_mat(labels, pred_probs, threshold, title="Confusion matrix (whole nucleus)"):
    """
    Plot a confusion matrix based on predicted probabilities and true labels.

    Args:
        labels (numpy.ndarray): True labels for each data point (0 for Young, 1 for Aged).
        pred_probs (numpy.ndarray): Predicted probabilities for each data point.
        threshold (float): The probability threshold to determine binary predictions.

    Returns:
        None
    """
    # Convert predicted probabilities to binary predictions using the threshold
    preds = (pred_probs >= threshold).astype(int)

    # Compute the confusion matrix
    conf_mat = tf.math.confusion_matrix(
        labels=labels, predictions=preds
    ).numpy()

    # Create a heatmap plot and configure parameters
    plt.figure(figsize=(4, 3))
    ax = sns.heatmap(conf_mat, annot=True, fmt="d", annot_kws={"fontsize":15}, cmap="Blues")

    ax.set_xlabel("Predicted Class", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(["Aged", "Young"])
    ax.set_ylabel("Real Class", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(["Aged", "Young"])
    ax.set_title(
        title, fontsize=14, pad=20
    )

    plt.show()


def plot_auc(labels, pred_probs, threshold):
    """
    Plot the Receiver Operating Characteristic (ROC) curve and AUC for
    binary classification.

    Args:
        labels (numpy.ndarray): True labels for each data point (0 for
            negative, 1 for positive).
        pred_probs (numpy.ndarray): Predicted probabilities for each
            data point.
        threshold (float): The probability threshold for determining
            positive class.

    Returns:
        None
    """

    # Compute the ROC curve and AUC
    fpr, tpr, t = roc_curve(labels, pred_probs)
    auc_score = auc(fpr, tpr)

    # Find the index of the decision threshold in the ROC curve
    idx_thresh = np.argmin(abs(threshold - t))

    # Create a plot
    plt.figure(figsize=(4, 3))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr, label=f"ROC curve (area = {auc_score:.3f})")

    # Plot vertical and horizontal lines at the selected threshold
    plt.axhline(y=tpr[idx_thresh], color="k", linestyle="--", lw=1)
    plt.axvline(x=fpr[idx_thresh], color="k", linestyle="--", lw=1)

    # Display the threshold value next to the point on the ROC curve
    plt.text(
        x=fpr[idx_thresh] - 0.1,
        y=tpr[idx_thresh] - 0.1,
        s=f"Threshold: {threshold:.2f}",
        fontsize=8,
    )

    # Mark the threshold point with a red circle
    plt.plot(fpr[idx_thresh], tpr[idx_thresh], "ro", markersize=4)

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="best")
    plt.show()




def plot_metrics(dataframes, dataset_names, thresholds):
    """
    Calculate and plot various classification metrics (Accuracy, Precision,
    Recall, AUC) for different datasets and thresholds.

    Args:
        dataframes (list of pandas.DataFrame): List of dataframes, each containing
            labels and predicted probabilities for a dataset.
        dataset_names (list of str): Names of the datasets corresponding to the dataframes.
        thresholds (list of float): List of probability thresholds to evaluate.

    Returns:
        pandas.DataFrame: A dataframe containing calculated metrics for each dataset
        and threshold.
    """
    # Initialize an empty list to store metric values
    thresh = []

    # List of metric names
    metrics_name = ["acc", "precision", "recall", "auc"]

    # Iterate over each threshold
    for t in thresholds:
        # Initialize a list of metric objects for the current threshold
        ms = [
            tf.keras.metrics.BinaryAccuracy(threshold=t),
            tf.keras.metrics.Precision(thresholds=t),
            tf.keras.metrics.Recall(thresholds=t),
            tf.keras.metrics.AUC(),
        ]

        # Iterate over dataset names and corresponding dataframes
        for ds_name, df in zip(dataset_names, dataframes):
            # Reset the state of all metric objects
            [m.reset_state() for m in ms]

            # Update metrics using true labels and predicted probabilities
            [m.update_state(df["label"], df["mean_prob"]) for m in ms]

            # Create a dictionary to store metric values along with other information
            d = {
                m_name: m.result().numpy()
                for m_name, m in zip(metrics_name, ms)
            }
            d = d | {"threshold": t, "dataset": ds_name}

            # Append the metric dictionary to the list
            thresh.append(d)

    # Create a pandas DataFrame from the list of metric dictionaries
    df = pd.DataFrame(thresh)

    # Calculate and add the F1-score to the DataFrame
    df["f1"] = (
        2 * (df["precision"] * df["recall"]) / (df["precision"] + df["recall"])
    )

    return df



def gradient_ascent_step(feature_extractor, img, filter_idx, lr):
    """
    Perform a single gradient ascent step to maximize the activation of a
    specific convolutional filter in a neural network.

    Args:
        feature_extractor (tf.keras.Model): The feature extractor model that
            contains the desired convolutional layer.
        img (tf.Tensor): The input image tensor to be modified.
        filter_idx (int): The index of the target convolutional filter.
        lr (float): The learning rate for gradient ascent.

    Returns:
        Tuple[float, tf.Tensor]: A tuple containing the loss value and the
        modified input image tensor.
    """
    with tf.GradientTape() as tape:
        tape.watch(img)
        activation = feature_extractor(img)

        # If the image is 3-dimensional (e.g., 3D convolution)
        if len(activation.shape) == 5:
            filter_activation = activation[:, 2:-2, 2:-2, 2:-2, filter_idx]
        # If the image is 2-dimensional (e.g., 2D convolution)
        elif len(activation.shape) == 4:
            filter_activation = activation[:, 2:-2, 2:-2, filter_idx]
        loss = tf.reduce_mean(filter_activation)

    # Compute and normalize the gradients
    grads = tape.gradient(loss, img)
    grads = tf.math.l2_normalize(grads)

    # Update the input image using the normalized gradients.
    img += lr * grads

    return loss, img


def visualize_filter(model, layer_name, filter_idx, img_shape):
    """
    Visualize the activation of a specific convolutional filter within a neural
    network layer by running gradient ascent.

    Args:
        model (tf.keras.Model): The neural network model.
        layer_name (str): The name of the target convolutional layer.
        filter_idx (int): The index of the target convolutional filter.
        img_shape (tuple): The shape of the input image (e.g., (height, width)).

    Returns:
        Tuple[float, np.ndarray]: A tuple containing the final loss value and
        the visualization of the filter activation as an image.
    """
    # Constant variables for gradient ascent iterations and learning rate
    iterations = 30
    lr = 10.0

    # Get the target layer and create a feature extractor model
    layer = model.get_layer(name=layer_name)
    feature_extractor = tf.keras.Model(
        inputs=model.inputs, outputs=layer.output
    )

    # Initialize empty tensor with random uniform noise
    img = tf.random.uniform((1, *img_shape, 1))

    # Run gradient ascent to visualize the filter activation
    for _ in range(iterations):
        loss, img = gradient_ascent_step(
            feature_extractor, img, filter_idx, lr
        )

    # Decode the resulting input image to make it interpretable
    img = deprocess_image(img[0].numpy())

    return loss, img


def deprocess_image(img):
    """
    Reverse the preprocessing steps applied to an image during neural network
    visualization to make it interpretable.
    """
    # Normalize array: center on 0., ensure variance is 0.15
    img = (img - img.mean()) / (img.std() + 1e-5)
    img *= 0.15

    # Center crop and clip to [0, 1]
    if len(img.shape) == 4:
        img = img[10:-10, 10:-10, 10:-10, :]
    elif len(img.shape) == 3:
        img = img[10:-10, 10:-10, :]
    img = np.clip(img, 0, 1)

    return img
