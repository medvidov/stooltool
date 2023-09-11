import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sk_metrics
from itertools import zip_longest
import csv
import os
import collections
from shutil import copyfile
from PIL import Image


def save_signed_diff_hitogram(
    truth: np.ndarray, predictions: np.ndarray, epoch_num: int, hist_output_dir: str
):
    """
    Save a histogram of the signed errors to the hist_output_dir.

    Inputs:
        truth: Numpy array of truth labels for the data
        predictions: Numpy array of the prediction for the data
        epoch_num: int of current epoch just completed
        hist_output_dir: string path to output directory in which to save histograms

    Output:
        None
    """
    # For each predictions, get signed error/difference
    signed_error = np.empty(truth.size, dtype=int)
    for i, (true_label, pred_label) in enumerate(zip_longest(truth, predictions)):
        signed_error[i] = pred_label - true_label

    # Plot the signed error data
    signed_error_sums = collections.Counter(signed_error)
    labels = signed_error_sums.keys()
    values = signed_error_sums.values()
    plt.bar(labels, values)

    # Label the axes
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("Signed Error")
    plt.ylabel("Number of Images")
    plt.title("Images Signed Error Epoch: " + str(epoch_num))
    plt.text(23, 45, r"$\mu=15, b=3$")

    # Save and close the histogram
    plt.savefig(
        os.path.join(hist_output_dir, "signed_error_hist_" + str(epoch_num) + ".png")
    )
    plt.close()


def create_one_hot(array: np.ndarray, num_items: int):
    """
    Create one hot array based on a truth array.

    Input:
        array: Numpy array of truth labels.
        num_items: int number of class labels in data (each one hot vector will be
                   this length)

    Output:
        one_hot: Numpy one hot array of input truth array. Shape: (len(array), num_items)
    """
    one_hot = np.zeros((array.size, num_items), dtype=int)
    one_hot[np.arange(array.size), array] = 1
    return one_hot


def binarize_array(array: np.ndarray, truth_value: int):
    """
    Binarize an array based on the truth_value. Elements which where equal to turth value
    will be one and all other elements will be zero.

    Input:
        array: Numpy array to be binarized.
        truth_values: int truth value to binarize the array on.

    Output:
        out: Numpy array of binarized input array based on truth_value.
    """
    out = np.zeros(array.size, dtype=int)

    for i in range(out.size):
        if array[i] == truth_value:
            out[i] = 1

    return out


def save_roc_curves(
    truth: np.ndarray,
    probabilities: np.ndarray,
    class_nums: np.ndarray,
    class_labels: np.ndarray,
    epoch_num: int,
    roc_output_dir: str,
):
    """
    Save ROC curves of each class label with AUC into roc_ouput_dir.

    Input:
        truth: Numpy array of truth labels.
        probabilities: Numpy array raw probabilities from prediction.
        class_num: Numpy array of int class labels based on number label.
        class_labels: Numpy array of actual string class labels.
        epoc_num: int of epoch just completed.
        roc_output_dir: string path to output directory in which ROC curves will be saved.

    Output:
        None
    """
    for label_num, label_name in zip_longest(class_nums, class_labels):
        # Binarize each array
        truth_binarized = binarize_array(truth, label_num)

        # Compute ROC curve
        fpr, tpr, _ = sk_metrics.roc_curve(
            truth_binarized, probabilities[:, label_num], drop_intermediate=False
        )

        # Find area under the curve
        roc_auc = sk_metrics.auc(fpr, tpr)

        # Make ROC curve plot
        plt.figure()
        plt.plot(fpr, tpr, color="red", label="ROC Curve Area: " + str(roc_auc))
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(label_name + " ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig(
            os.path.join(
                roc_output_dir, label_name + "_roc_fig_" + str(epoch_num) + ".png"
            )
        )
        plt.close()


def get_truth_predictions_for_label(truth: np.ndarray, predictions: np.ndarray, label: str):
    """
    Get the truth labels and predicted scores that correspond the a certain truth label

    Input:
        truth: Numpy array of truth labels
        predictions: Numpy array of prediction scores for each image
        label: string of label we want predictions for

    Output:
        truth: Numpy array of truth labels of type label
        predictions: Numpy array of prediction scores coresponding to truth of type label
    """
    # Get indices of label in truth
    indices = np.where(truth == label)

    # Return truth and predictions arrays with indices
    return truth[indices], predictions[indices]


def root_mean_squared_error(truth_labels: np.ndarray, predicted_labels: np.ndarray):
    """
    Get the root mean squared error of the provided truth and predicted labels/
    
    Input:
        truth_labels: Numpy array of truth labels for each image
        predicted_labels: Numpy array of predicted labels for each image

    Output:
        Numpy array of root-mean-squared-error of the truth and predicted labels for each image
    """
    if len(truth_labels) != len(predicted_labels):
        raise RuntimeError(
            "truth_labels and predicted_labels do not have the same length."
        )

    return np.sqrt(
        np.sum(np.square(predicted_labels - truth_labels)) / len(truth_labels)
    )


def root_mean_squared_error_average_each_label(
    truth_labels: np.ndarray, predicted_labels: np.ndarray, class_labels: list
):
    """
    Get the root-mean-squared-error and accuracies accross each label.

    Input:
        truth_labels: Numpy array of truth labels for each image
        predicted_labels: Numpy array of predicted labels for each image
        class_labels: List of class labels used during training

    Output
        rmse: Numpy array of root-mean-squared-error for each label
        accuracies: Numpy array of accuracies for each label 
    """
    rmse, accuracies = np.empty(len(class_labels), dtype=float), np.empty(
        len(class_labels), dtype=float
    )

    for i, label in enumerate(class_labels):
        # Get cases corresponding to this label
        truth, predicted = get_truth_predictions_for_label(
            truth_labels, predicted_labels, label
        )

        # Get RMSE acore for this label
        rmse[i] = root_mean_squared_error(truth, predicted)

        # Get accuracy score for this label
        accuracies[i] = sk_metrics.accuracy_score(truth, predicted)

    return rmse, accuracies


def save_confusion_matrix(
    test_labels: np.ndarray, predictions: np.ndarray, class_nums: int, 
    class_labels: list, epoch_num: int, output_dir: str
):
    """
    Compute confusion matrix and normalize. Save confusion matrix to directory.

    Input:
        test_labels: Numpy array of test labels
        predictions: Numpy array of prediction scores for each label for each image
        class_nums: int number of class labels used in training
        class_labels: list of class labels used in training
        epoch_num: int epoch number just trained on
        output_dir: string of output directory in which to save confusion matrix

    Output:
        None
    """
    # Compute confusion matrix
    confusion = sk_metrics.confusion_matrix(test_labels, predictions, labels=class_nums)
    axis_labels = class_labels
    sns.heatmap(
        confusion,
        xticklabels=axis_labels,
        yticklabels=axis_labels,
        cmap="Blues",
        annot=True,
        fmt="g",
        square=True,
    )
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(os.path.join(output_dir, "cm_" + str(epoch_num) + ".png"))
    plt.close()


def save_csv_metrics(
    test_labels: np.ndarray, predictions: np.ndarray, class_nums: int, 
    class_labels: list, epoch_num: int, output_dir: str
):
    """
    Compute precision, recall, F1, RMSE, and accuracy metrics for each class label.
    Save metrics csv to directory.

    Input:
        test_labels: Numpy array of test labels
        predictions: Numpy array of prediction scores for each label for each image
        class_nums: int number of class labels used in training
        class_labels: list of class labels used in training
        epoch_num: int epoch number just trained on
        output_dir: string of output directory in which to save metrics csv

    Output:
        None
    """
    # Calculte precision, recall, and f1 score
    precision, recall, f1, _ = sk_metrics.precision_recall_fscore_support(
        test_labels, predictions, labels=class_nums
    )

    # Calculate the RMSE and accuracy for each class label
    rmse, accuracy = root_mean_squared_error_average_each_label(
        test_labels, predictions, class_nums
    )

    # Write out metrics per class label to csv
    with open(
        os.path.join(output_dir, "class_metrics_" + str(epoch_num) + ".csv"), "w"
    ) as metrics_csv:
        # Get writer
        metrics_csv_writer = csv.writer(metrics_csv)

        # Write header
        metrics_csv_writer.writerow(
            ["Labels", "Precision", "Recall", "F1", "RMSE", "Accuracy"]
        )

        # Write metrics for each class label
        for (
            class_label,
            precision_score,
            recall_score,
            f1_score,
            rmse_score,
            accuracy_score,
        ) in zip_longest(class_labels, precision, recall, f1, rmse, accuracy):

            # Write out metrics for this class label
            metrics_csv_writer.writerow(
                [
                    class_label,
                    precision_score,
                    recall_score,
                    f1_score,
                    rmse_score,
                    accuracy_score,
                ]
            )

    # Calculate overall metrics
    precision, recall, f1, _ = sk_metrics.precision_recall_fscore_support(
        test_labels, predictions, labels=class_nums, average="micro"
    )
    rmse = root_mean_squared_error(test_labels, predictions)
    accuracy = sk_metrics.accuracy_score(test_labels, predictions)

    # Calculate averaged overall metrics
    (
        precision_averaged,
        recall_average,
        f1_averaged,
        _,
    ) = sk_metrics.precision_recall_fscore_support(
        test_labels, predictions, labels=class_nums, average="macro"
    )
    accuracy_averaged = sk_metrics.balanced_accuracy_score(test_labels, predictions)

    # Write out overall metrics
    with open(
        os.path.join(output_dir, "metrics_" + str(epoch_num) + ".csv"), "w"
    ) as metrics_csv:
        # Get writer
        metrics_csv_writer = csv.writer(metrics_csv)

        # Write header
        metrics_csv_writer.writerow(
            [
                "Precision",
                "Averaged Precision",
                "Recall",
                "Averaged Recall",
                "F1",
                "Averaged F1",
                "RMSE",
                "Accuracy",
                "Averaged Accuracy",
            ]
        )

        # Write metrics
        metrics_csv_writer.writerow(
            [
                precision,
                precision_averaged,
                recall,
                recall_average,
                f1,
                f1_averaged,
                rmse,
                accuracy,
                accuracy_averaged,
            ]
        )


def compute_save_metrics(
    test_labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, 
    class_labels: list, epoch_num: int, output_dir: str
):
    """
    Compute confusion matrix, class-wise and averaged precision, recall, F1, RMSE, and accuracy metrics,
    and ROC curve for each class label.

    Input:
        test_labels: Numpy array of test labels
        predictions: Numpy array of prediction labels for each image
        probabilities: Numpy array of probabilities for each label for each image
        class_nums: int number of class labels used in training
        class_labels: list of class labels used in training
        epoch_num: int epoch number just trained on
        output_dir: string of output directory in which to save metrics csv
    """
    # Make the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert test truth labels from one hot to index
    test_labels = np.argmax(test_labels, axis=1)

    # Make confusion matrix
    cm_output_dir = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(cm_output_dir, exist_ok=True)
    save_confusion_matrix(
        test_labels,
        predictions,
        list(class_labels),
        list(class_labels.values()),
        epoch_num,
        cm_output_dir,
    )

    # Sve CSV of precision, recall, f1, RMSE, and accuracy
    csv_output_dir = os.path.join(output_dir, "metrics")
    os.makedirs(csv_output_dir, exist_ok=True)
    save_csv_metrics(
        test_labels,
        predictions,
        list(class_labels),
        list(class_labels.values()),
        epoch_num,
        csv_output_dir,
    )

    # Save ROC curves for all class labels
    roc_output_dir = os.path.join(output_dir, "roc_curves")
    os.makedirs(roc_output_dir, exist_ok=True)
    save_roc_curves(
        test_labels,
        probabilities,
        list(class_labels),
        list(class_labels.values()),
        epoch_num,
        roc_output_dir,
    )

    # Save histogram of signed differance on images
    signed_diff_hist_dir = os.path.join(output_dir, "signed_diff_hists")
    os.makedirs(signed_diff_hist_dir, exist_ok=True)
    save_signed_diff_hitogram(test_labels, predictions, epoch_num, signed_diff_hist_dir)


def save_false_positives(
    test_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_labels: collections.OrderedDict,
    paths: np.ndarray,
    output_dir: str,
):
    """
    Saves false positives for each class label to an output directory.

    Input:
        test_labels: Numpy array of all truth labels.
        pred_labels: Numpy array of all predicted labels.
        class_labels: Collections OrderedDict object of key to label pairings
        paths: Numpy array of string paths to each image
        output_dir: String path to output directory in which to save all false positives

    Output:
        None
    """
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get indices of the false positives
    false_positives = {label: list() for label in class_labels.values()}
    for index, (truth_label, pred_label) in enumerate(
        zip_longest(test_labels, pred_labels)
    ):
        if truth_label != pred_label:
            false_positives[class_labels[truth_label]].append(index)

    # Save out all of the false positives
    for label, indices in false_positives.items():
        # Create subdirectory for this label
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

        # Copy file path over
        for index in indices:
            path = paths[index]
            copyfile(path, os.path.join(output_dir, label, os.path.split(path)[-1]))

def compute_segnet_metrics(test_labels: np.ndarray, predictions: np.ndarray, class_nums: list, class_labels: list, test_uuids: list, test_images: np.ndarray, epoch_num: int, output_dir: str):
    # Round predictions to whether or not pixel is stool
    predictions = np.round(predictions)
    
    # Calculate the mean intersection over untion of each image
    csv_output_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(csv_output_dir, exist_ok=True)
    
    # Add truth and predictions together
    combination = test_labels + predictions

    # Intersection and union
    intersection = np.where(combination == 2, 1, 0)
    union = np.where(combination >= 1, 1, 0)

    intersection = np.count_nonzero(intersection == 1, axis=(1,2))
    union = np.count_nonzero(union == 1, axis=(1,2))

    # Calculate IoU
    ious = intersection / union
    mean_iou = np.average(ious)

    # Write out overall metrics
    with open(os.path.join(csv_output_dir, 'metrics_' + str(epoch_num) + '.csv'), 'w') as metrics_csv:
        # Get writer
        metrics_csv_writer = csv.writer(metrics_csv)

        # Write header
        metrics_csv_writer.writerow(
            ['uuid', 'iou']
        )

        # Write metrics
        for uuid, iou in zip_longest(test_uuids, ious):
            metrics_csv_writer.writerow([uuid, iou])

        # Write mean iou
        metrics_csv_writer.writerow(['mean', mean_iou])

    image_dir = os.path.join(output_dir, 'images', str(epoch_num))
    os.makedirs(image_dir, exist_ok=True)
    # Save test images and truth/predicted masks
    for test, uuid in zip_longest(test_images, test_uuids):
        image = Image.fromarray((test*225).astype(np.uint8))
        image.save(os.path.join(image_dir, uuid + '.jpg'))
    pred_images = np.where(predictions == 1, [255, 255, 255], [0, 0, 0]).astype(np.uint8)
    for pred, uuid in zip_longest(pred_images, test_uuids):
        image = Image.fromarray(pred)
        image.save(os.path.join(image_dir, uuid + '_pred_mask.jpg'))
    truth_images = np.where(test_labels == 1, [255, 255, 255], [0, 0, 0]).astype(np.uint8)
    for truth, uuid in zip_longest(truth_images, test_uuids):
        image = Image.fromarray(truth)
        image.save(os.path.join(image_dir, uuid + '_truth_mask.jpg'))
