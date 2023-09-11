import os
import sys
import tensorflow as tf
import numpy as np
from utils import (
    get_train_validation_sets,
    decode_and_resize_images,
    get_batch,
    get_images_and_labels,
    get_configs,
    augment_image,
    augment_all_images,
    downsample_train_data
)
from model import get_classifier
from metrics import compute_save_metrics, save_false_positives
from segnet import train_segnet
from shutil import copyfile


def main(config_path):
    # Set-up memory usage
    # Code from tensorflow 2 documentation: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)

    # Get configs
    configs = get_configs(config_path)

    # Get results directory
    results_dir = configs["results_dir"]

    # Perform data augmentation on all images
    augment_all_images(configs["setup_params"]["training_data_dir"])

    # Get train and test sets and classes
    train, test, classes = get_train_validation_sets(configs["setup_params"])

    # Downsample the training data
    if configs["setup_params"]["downsample_train_data"]:
        train = downsample_train_data(train, classes)

    # Preprocessing images into tensors with size expected by the image module.
    input_dimensions = configs['training_params']['image_dimensions']
    train_images, train_labels, test_images, test_labels = decode_and_resize_images(
        train,
        test,
        classes,
        input_dimensions
    )

    # TODO: Temporarily make validation set larger to be multiple of 16
    test = np.append(test, test[:12], axis=0)
    test_images = np.append(test_images, test_images[:12], axis=0)
    test_labels = np.append(test_labels, test_labels[:12], axis=0)

    # If we want to do image segmentation
    if (configs['training_params']['use_image_segmentation']):
        # Do image segmentation training and crop training data with masks
        train_images, test_images = train_segnet(
            configs['training_params'], 
            train, 
            test, 
            train_images, 
            test_images, 
            os.path.join(results_dir, 'segnet')
        )

    # Load a pre-trained TF-Hub module for extracting features from images.
    model = get_classifier(configs['training_params'], len(classes))

    # Create model save directory
    model_save_dir = os.path.join(results_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)

    # Save config file and git hash to results directory
    copyfile(config_path, os.path.join(results_dir, "config.yaml"))
    git_info_file = os.path.join(results_dir, "git.txt")
    os.system("git rev-parse HEAD > " + git_info_file)
    os.system("git diff >> " + git_info_file)

    # Training loop
    false_pos_dir = os.path.join(results_dir, "false_positives")
    num_epochs = configs["training_params"]["num_epochs"]
    for i in range(num_epochs):
        # Print the status of training
        print("Training on epoch num " + str(i+1) + "/" + str(num_epochs))

        # Train on training data and update weights
        model.fit(
            train_images, 
            train_labels, 
            batch_size=configs['training_params']["batch_size"]
        )

        # Predict on validation/test images
        predictions = model.predict(test_images)

        # Convert predictions to labels and compute metrics
        pred_labels = np.argmax(predictions, axis=1)
        compute_save_metrics(
            test_labels,
            pred_labels,
            predictions,
            classes,
            i + 1,
            results_dir,
        )

        # Save false positives
        save_false_positives(
            np.argmax(test_labels, axis=1),
            pred_labels,
            classes,
            test[:, 0],
            os.path.join(false_pos_dir, str(i + 1)),
        )

        # Save model and weights
        model.save(os.path.join(model_save_dir, "model_" + str(i + 1)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py [TRAINING CONFIG YAML PATH]")
        sys.exit(0)
    main(sys.argv[1])