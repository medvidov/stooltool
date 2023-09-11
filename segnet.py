from metrics import compute_segnet_metrics
from utils import crop_image_from_mask
from itertools import zip_longest
from collections import OrderedDict
from model import get_segnet
import tensorflow as tf
import numpy as np
import collections
import pickle
import os

def path_to_uuid(path_array: np.ndarray):
    """
    Converts from and array of file path, label pairings to just the filename for an image.

    Input:
        path_array: Numpy array of tuples of image file paths and labels

    Output:
        converted: numpy array of image file names
    """
    split = np.char.split(path_array[:,0], '/')
    converted = np.empty(split.shape, dtype=object)
    for i, elem in enumerate(split):
        converted[i] = elem[-1].split('.')[0]
    return converted

def train_segnet(configs: dict, train_paths: np.ndarray, test_paths: np.ndarray, 
    train_images: np.ndarray, test_images: np.ndarray, results_dir: str):
    """
    Set-up truth data for training SegNet, train SegNet, and then perform a prediction to get stool for use in bristol classifier

    Input:
        configs: Dictionary of training configurations
        train_paths: Numpy array of paths to the training images
        test_paths: Numpy array of paths to the test images
        train_images: Numpy array of training images
        test_imagse: Numpy array of test images
        results_dir: Path to directory were results will be entered

    Output:
        images: Numpy array training images just containing portions SegNet classiified as containing Stool
    """
    # Get the order of the train and test sets
    train_order = path_to_uuid(train_paths)
    test_order = path_to_uuid(test_paths)
    
    # Load the dict of truth data for SegNet
    truth = None
    with open(configs['image_segmentaiton']['truth_pickle'], 'rb') as file:
        truth = pickle.load(file)
    
    # Create ordered truth
    train_truth_objects = np.array([truth[key] for key in train_order])
    test_truth_objects = np.array([truth[key] for key in test_order])
    # Crop truth masks in the same way the training/test data was cropped
    # Currently, just crop in center
    # TODO: Can't use ragged tensors because they're buggy
    input_dimensions = tuple(configs['image_dimensions'])
    # size = tf.convert_to_tensor(
    #     np.full((len(train_truth), 2), input_dimensions, dtype=(int, int)), 
    #     dtype=tf.int32
    # )
    # train_truth = tf.map_fn(
    #     tf.image.resize_with_crop_or_pad, 
    #     (tf.ragged.constant(train_truth), size[:,0], size[:,1]),  
    #     dtype=tf.float32
    # )
    # import pdb; pdb.set_trace()
    # size = tf.convert_to_tensor(
    #     np.full((len(test_truth), 2), input_dimensions, dtype=(int, int)), 
    #     dtype=tf.int32
    # )
    # test_truth = tf.map_fn(
    #     tf.image.resize_with_crop_or_pad, 
    #     (tf.ragged.constant(test_truth), size[:,0], size[:,1]),
    #     dtype=tf.float32
    # )
    # TODO make this faster
    train_truth = np.empty(train_truth_objects.shape + input_dimensions + (1,))
    test_truth = np.empty(test_truth_objects.shape + input_dimensions + (1,))
    for i in range(len(train_truth_objects)):
        train_truth[i] = tf.image.resize_with_crop_or_pad(train_truth_objects[i], input_dimensions[0], input_dimensions[1])
    for i in range(len(test_truth_objects)):
        test_truth[i] = tf.image.resize_with_crop_or_pad(test_truth_objects[i], input_dimensions[0], input_dimensions[1])

    # Compile SegNet
    segnet = get_segnet(configs)
    
    # Train on Segnet
    num_epochs = configs['num_epochs']
    batch_size = configs["batch_size"]
    model_save_dir = os.path.join(results_dir, 'models')
    false_pos_dir = os.path.join(results_dir, "false_positives")
    print('Training on SegNet')
    for i in range(num_epochs):
        # Print the status of training
        print("Training on epoch num " + str(i+1) + "/" + str(num_epochs))

        # Train on training data and update weights
        segnet.fit(
            train_images, 
            train_truth, 
            batch_size=batch_size
        )

        # Predict on validation/test images
        predictions = segnet.predict(test_images)

        # Compute Metrics
        compute_segnet_metrics(test_truth, predictions, [1], ['stool'], test_order, test_images, i+1, results_dir)

        # Save model and weights
        segnet.save(os.path.join(model_save_dir, "model_" + str(i+1)))

    # Cut out masks from the training and test images
    # TODO: Better Performance
    # train_images = tf.map_fn(crop_image_from_mask, (train_images, train_truth))
    # test_images = tf.map_fn(crop_image_from_mask, (test_images, test_truth))
    train_images = np.array([crop_image_from_mask((image, mask)) for image, mask in zip_longest(train_images.numpy(), train_truth)])
    test_images = np.array([crop_image_from_mask((image, mask)) for image, mask in zip_longest(test_images, test_truth)])
    
    return train_images, test_images