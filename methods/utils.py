import numpy as np
import seaborn as sns
import tensorflow as tf
import re
import matplotlib.pyplot as plt
from scipy import stats
import sklearn.metrics as skmetrics
from sklearn.calibration import calibration_curve


def get_class_palette():
    
    colors = sns.color_palette("colorblind")
    colors2 = sns.color_palette("tab20")
    pal = {
        "young": colors[2],
        "aged": colors[3],
        "aged_CASIN": colors[0],
        "aged_DMSO": colors[4],
        "aged_IOX": colors[5],
        "aged_UNC": colors[6],
        "aged_treated_RhoAi": colors[7],
        "middle": colors[8],
        "myeloid_progenitors": colors[9],
        "young_treated_NaB": colors2[8],
    }
    
    return pal


def get_microscopist_palette():
    
    colors = sns.color_palette("colorblind")
    pal = {
        "Microscopist 1": colors[1],
        "Microscopist 2": colors[9],
        "Microscopist 3": colors[4],
    }
    
    return pal


def fix_repeated_metric_names(history):
    # Regular expression pattern to match an underscore followed by a number
    pattern = r"_\d+"

    rename_dic = {}

    # Iterate through the metrics and modify the matching elements
    for k in history.keys():
        if re.search(pattern, k):
            # Replace the matched substring with an empty string
            new_k = re.split(pattern, k)[0]
            rename_dic[k] = new_k

    new_hist = {rename_dic.get(k, k): v for k, v in history.items()}

    return new_hist


def get_id_from(files, level="nucleus", extension="png"):
    """
    Extract unique nucleus identifiers from a list of file names.

    This function searches for file names containing a pattern "_nuc" and
    extracts unique nucleus identifiers by removing the "_nuc" suffix.

    Args:
        files (list of str): A list of file names to process.

    Returns:
        list of str: A list of unique nucleus identifiers.
    """
    # Regular expression pattern to match
    if level == "nucleus":
        pattern = r"_\d+\." + extension
    elif level == "batch":
        pattern = r"_nuc"
    else:
        print("level argument must one either nucleus or batch")

    new_names = []

    # For each file name
    for f in files:
        # If it matches the pattern "_nuc"
        if re.search(pattern, f):
            # Split the file name at the "_nuc" pattern and keep the first part
            new_names.append(re.split(pattern, f)[0])

    # Convert the list to a set to remove duplicates, then back to a list
    return new_names


def get_unique_from(files, level="nucleus"):
    """
    Extract unique nucleus identifiers from a list of file names.

    This function searches for file names containing a pattern "_nuc" and
    extracts unique nucleus identifiers by removing the "_nuc" suffix.

    Args:
        files (list of str): A list of file names to process.

    Returns:
        list of str: A list of unique nucleus identifiers.
    """
    # Regular expression pattern to match
    if level == "nucleus":
        pattern = r"_\d+\.png"
    elif level == "batch":
        pattern = r"_nuc"
    else:
        print("level argument must one either nucleus or batch")

    new_names = []

    # For each file name
    for f in files:
        # If it matches the pattern "_nuc"
        if re.search(pattern, f):
            # Split the file name at the "_nuc" pattern and keep the first part
            new_names.append(re.split(pattern, f)[0])

    # Convert the list to a set to remove duplicates, then back to a list
    return list(set(new_names))



def get_nuc_Z_position(nuc_ids):
    
    pattern = r"\d+\.png"
    
    nuc_pos = [re.search(pattern, p) for p in nuc_ids]
    # Remove the .png extension
    nuc_pos = [int(p.group()[:-4]) for p in nuc_pos]
    return nuc_pos


def get_max_size(npz_list, n_dim=3):
    """
    Get the maximum size (shape) among a list of Numpy arrays.

    This function iterates through a list of Numpy arrays stored in .npz files,
    extracts their shapes, and returns the maximum size among all arrays.

    Args:
        npz_list (list of str): A list of file paths to .npz files containing Numpy arrays.
        n_dim (int): The number of dimensions expected in the arrays (default is 3).

    Returns:
        tuple of int: A tuple representing the maximum size of the arrays.
    """
    # Initialize an empty array to store sizes
    array_sizes = np.empty(shape=(len(npz_list), n_dim))

    # Iterate through the .npz files
    for i, nuc_path in enumerate(npz_list):
        img = np.load(nuc_path)["img"]
        array_sizes[i] = img.shape

    # Find the maximum size among all arrays
    max_size = array_sizes.max(axis=0)

    # Convert the resulting tuple of floats to integers
    max_size = tuple(int(x) for x in max_size)

    return max_size



def get_n_duplicates(lst):
    """Return the number of duplicates found in a list of elements"""
    return len(lst) - len(set(lst))


def get_common(l1, l2):
    """Return the common elements between two lists"""
    return list(set(l1).intersection(l2))


def make_batch(input_list, batch_size):
    grouped_lists = []
    for i in range(0, len(input_list), batch_size):
        group = input_list[i : i + batch_size]
        grouped_lists.append(group)
    return grouped_lists


def dataset_from_partition(
    imgs_dir,
    partition,
    channel_mode,
    label_mode,
    image_size,
    batch_size,
    seed=2023,
):
    ds = tf.keras.utils.image_dataset_from_directory(
        f"{imgs_dir}/{partition}/",
        color_mode=channel_mode,
        labels="inferred",
        label_mode=label_mode,
        interpolation="bilinear",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    nuc_ids = get_id_from(ds.file_paths, level="nucleus")
    nuc_ids = [s.split("/")[-1] for s in nuc_ids]
    if batch_size is not None and batch_size > 1:
        nuc_ids = make_batch(nuc_ids, batch_size=batch_size)

    return nuc_ids, ds


def dataset_from_partition_cv(
    imgs_dir,
    folds,
    channel_mode,
    label_mode,
    image_size,
    batch_size,
    seed=2023,
):
    ds = None
    all_nuc_ids = []

    for fold in folds:
        nuc_ids, this_ds = dataset_from_partition(
            imgs_dir,
            fold,
            channel_mode,
            label_mode,
            image_size,
            batch_size,
            seed=2023,
        )

        all_nuc_ids.extend(nuc_ids)
        ds = this_ds if ds is None else ds.concatenate(this_ds)

    return all_nuc_ids, ds




def confidence_interval(n, mean, std, conf_level=0.95):
      
    # Calculate the confidence interval using the normal approximation
    z_value = stats.norm.ppf((1 + conf_level)/2)  # Z-value for 95% CI (≈ 1.96)
    
    # Standard error
    std_error = std / np.sqrt(n)
    
    # Confidence interval
    ci_lower = mean - z_value * std_error
    ci_upper = mean + z_value * std_error

    return (float("{:.2f}".format(ci_lower)),
           float("{:.2f}".format(ci_upper)))



def plot_distribution_comparison(plot_df, pal):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Create FacetGrid
    g = sns.FacetGrid(
        plot_df, row="condition", hue="condition", 
        aspect=6, height=0.75, palette=pal, 
        sharey=False
    )

    # Draw the densities
    g.map(sns.kdeplot, "mean_prob", bw_adjust=.5, fill=True, alpha=0.8, lw=1.5)
    g.map(sns.kdeplot, "mean_prob", color="k", lw=2, bw_adjust=.5)

    # Reference line at y=0
    g.refline(y=0, linewidth=2, linestyle="-", color="black", clip_on=False)

    # Function to label each subplot
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0., 1., label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    
    g.map(label, "mean_prob")

    # Adjust layout to prevent overlap
    g.figure.subplots_adjust(hspace=0.3)

    # Modify each subplot
    for i, ax in enumerate(g.axes.flat):
        ax.yaxis.tick_right()  # Move ticks to the right
        ax.spines["left"].set_visible(False)  # Hide left spine
        ax.spines["right"].set_visible(True)  # Keep right spine visible
        ax.spines["right"].set_color("black")  # Ensure visibility
        
        # Remove y-labels for all but the last plot
        if i < len(g.axes.flat) - 1:
            ax.set_ylabel("")
    
    # Set the y-axis label only on the last (bottom-most) subplot
    g.axes[-1, 0].set_ylabel("Density", fontsize=12, labelpad=15)
    g.axes[-1, 0].yaxis.set_label_position("right")  # Move y-axis label to the right

    # Clean up aesthetics
    g.set_titles("")
    g.set(xlim=(-0.1, 1.1))
    g.set_axis_labels("p(young)")
    
    return g
    


def annotate(ax, data, x, y):
    _, _, rvalue, pvalue, _ = stats.linregress(x=data[x], y=data[y])
    ax.text(.02, .9, f'r2={rvalue ** 2:.2f}, p={pvalue:.2g}', transform=ax.transAxes)



def plot_regplot(df, x, y, hue, y_label, palette, legend=False):
    
    plt.figure(figsize=(4, 3))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette=palette, legend=legend)
    ax = sns.regplot(data=df, x=x, y=y,
        scatter=False, truncate=False, order=1, color=".2",
    )
    annotate(ax, data=df, x=x, y=y)
    plt.xlabel("p(young)")
    plt.ylabel(y_label)



def plot_multiple_auc(labels, pred_probs, legends, thresholds):
    """
    Plot the Receiver Operating Characteristic (ROC) curve and AUC for
    binary classification.

    Args:
        threshold (float): The probability threshold for determining
            positive class.

    Returns:
        None
    """

    plt.figure(figsize=(4, 3))
    plt.plot([0, 1], [0, 1], "k--")
    
    for lab, pr, leg, t in zip(labels, pred_probs, legends, thresholds):
        # Compute the ROC curve and AUC
        fpr, tpr, auc_thresholds = skmetrics.roc_curve(lab, pr)
        auc_score = skmetrics.auc(fpr, tpr)
    
        # Find the index of the decision threshold in the ROC curve
        idx_thresh = np.argmin(abs(t - auc_thresholds))
    
        # Create a plot
        plt.plot(fpr, tpr, label=f"{leg} (AUC {auc_score:.3f})")
        # Mark the threshold point with a red circle
        plt.plot(fpr[idx_thresh], tpr[idx_thresh], "ro", markersize=6)

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="best")
    plt.show()



def plot_calibration_curve(y, probs, legends):

    # Plot the calibration curve
    plt.figure(figsize=(4, 3))
    
    for pr, leg in zip(probs, legends):
        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(y, pr, n_bins=10, strategy='uniform')
        
        plt.plot(prob_pred, prob_true, marker='o', label=leg)
        
    plt.plot([0, 1], [0, 1], linestyle='--', label="Perfectly calibrated")
    plt.title('Calibration plot')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.legend()
    plt.show()
