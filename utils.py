import tensorflow as tf
import numpy as np
import collections
import random
import os
import yaml
from PIL import Image
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    array_to_img,
    img_to_array,
    load_img,
)


def get_configs(config_path: str):
    """
    Get settings based on the contents of the specified config file.

    Input:
        config_path: String containing the config file path

    Output:
        configs: Dict of configs
    """

    # Reads in the config file for future use
    configs = {}
    try:
        with open(config_path, "r") as config_file:
            configs = yaml.safe_load(config_file)
    except Exception as e:
        print(e)
    return configs


def make_train_and_test_sets(
    images_dir: str, train_fraction: float = 0.8, random_seed: int = 0
):
    """
    Split the data into train and test sets and get the label classes.

    Input:
        images_dir: String containing the image directory file path
        train_fraction: The percent (expressed as a decimal) of images that will be used for trainig
        random_seed: Random seed to ensure we can replicate image split

    Output:
        train_examples: Array of training data images
        test_examples: Array of testing data images
        classes: Bristol scale rating of all images
    """
    train_examples, test_examples = [], []
    shuffler = random.Random(random_seed)
    is_root = True
    classes = None
    for (dirname, subdirs, filenames) in tf.io.gfile.walk(images_dir):
        # The root directory gives us the classes
        if is_root:
            subdirs = sorted(subdirs)
            classes = collections.OrderedDict(enumerate(subdirs))
            label_to_class = dict([(x, i) for i, x in enumerate(subdirs)])
            is_root = False
        # The sub directories give us the image files for training.
        else:
            filenames.sort()
            shuffler.shuffle(filenames)
            full_filenames = [os.path.join(dirname, f) for f in filenames]
            _, label = os.path.split(dirname)
            label_class = label_to_class[label]

            # An example is the image file and it's label class.
            examples = list(zip(full_filenames, [label_class] * len(filenames)))
            num_train = int(len(filenames) * train_fraction)
            train_examples.extend(examples[:num_train])
            test_examples.extend(examples[num_train:])

    shuffler.shuffle(train_examples)
    shuffler.shuffle(test_examples)
    return np.array(train_examples), np.array(test_examples), classes


def make_separate_train_test_sets(train_dir: str, test_dir: str, random_seed: int = 0):
    """
    Splits data into separate training and testing sets.

    Input:
        train_dir: String containing the training folder file path
        test_dir: String containing the testing folder file path
        random_seed: Random seed to ensure we can replicate image split

    Output:
        train_data: Model training data
        test_data: Model testing data
        classes: Bristol scale rating of all images
    """
    # Get training data
    train_data, _, classes = make_train_and_test_sets(train_dir, 1.0, random_seed)

    # Get test (validation) data
    test_data, _, _ = make_train_and_test_sets(test_dir, 1.0, random_seed)

    return train_data, test_data, classes


def get_train_validation_sets(configs: dict):
    """
    Gets and uses the validation data if the config file specifies a validation directory.

    Input:
        configs: Dict containing config information

    Output:
        Either returns separate train and test sets or uses a training/test split
    """

    # Creates train, test, and validation sets
    if configs["use_validation_dir"]:
        return make_separate_train_test_sets(
            "augmented_data", configs["validation_data_dir"], configs["seed"]
        )

    # Creates train and test sets
    return make_train_and_test_sets(
        "augmented_data", configs["train_fraction"], configs["seed"]
    )


def get_label(example):
    """
    Get the label (number) for given example.

    Input:
        example: A piece of sample data (image and label class)

    Output:
        That exmaple's label class (Bristol scale rating)
    """
    return example[1]


def get_encoded_image(image_path: str):
    """
    Get the image data (encoded jpg) of given example.

    Input:
        image_path: A string path to image to be decoded
    """
    return tf.io.gfile.GFile(image_path, "rb").read()

def decode_and_resize_images(train_data: list, test_data: list, classes: collections.OrderedDict, input_dimensions: list):
    """
    Decode and resize the input images differently based on CNN used

    Inputs:
        train_data: List of training images in bytes
        test_data: List of test images in bytes
        classes: OrderedDict of classes used in this network
        input_dimensions: List of size 2 denoting the dimensions of input images to the network
    """
    # Convert the training data
    train_images, train_labels = get_images_and_labels(train_data, len(classes))
    size = tf.convert_to_tensor(
        np.full((len(train_images), 2), input_dimensions, dtype=(int, int)),
        dtype=tf.int32,
    )
    train_images = tf.map_fn(
        decode_and_resize_image,
        (tf.convert_to_tensor(train_images), size),
        dtype=tf.float32,
    )

    # Convert the testing data
    test_images, test_labels = get_images_and_labels(test_data, len(classes))
    size = tf.convert_to_tensor(
        np.full((len(test_images), 2), input_dimensions, dtype=(int, int)),
        dtype=tf.int32,
    )
    test_images = tf.map_fn(
        decode_and_resize_image,
        (tf.convert_to_tensor(test_images), size),
        dtype=tf.float32,
    )

    return train_images, np.array(train_labels), test_images, test_labels


def decode_and_resize_image(args):
    """
    Decodes images from bytes and crops images in the center
    Inputs to this function are weird due to use of tf.map_fn in train.py

    Input:
        args: tuple of (bytes string of image data, tensor of crop dimensions)

    Output:
        Decoded and cropped image
    """
    # Parse args
    encoded = args[0]
    image_dimensions = args[1].numpy()

    # Decode image
    decoded = tf.image.decode_jpeg(encoded, channels=3)
    decoded = tf.image.convert_image_dtype(decoded, tf.float32)

    # Return cropped image
    return tf.image.resize_with_crop_or_pad(
        decoded, image_dimensions[0], image_dimensions[1]
    )


def get_batch(images: np.ndarray, labels: np.ndarray, batch_size=None):
    """
    Get a random batch of examples.

    Input:
        images: Array of images
        labels: Array of labels corresponding to images
        batch_size: The number of images that will be in the batch

    Output:
        Images and corresponding labels in the batch
    """

    # If there is batch size, create the batch
    if batch_size:
        idx = np.random.choice(images.shape[0], batch_size, replace=False)
        return images[idx], labels[idx]
    return images, labels


def get_images_and_labels(batch_examples, num_classes):
    """
    Get the images and corresponding labels for use.

    Input:
        batch_examples: Batch of examples we want images and labels of
        num_classes: Number of available classes to label

    Output:
        Images and one hot labels from that batch.
    """

    # Creates arrays of images and labels for those images
    images = [get_encoded_image(e[0]) for e in batch_examples]
    classes = []
    for i in range(num_classes):
        classes.append(i + 1)
    one_hot_labels = [get_label_one_hot(e, num_classes) for e in batch_examples]
    return images, one_hot_labels


def get_label_one_hot(example, num_classes):
    """
    Get the one hot encoding vector for the example.

    Input:
        example: A single image and label pair
        num_classes: Number of classes available to label
    Output:
        Vector containing one hot label for the example
    """

    # Creates one hot encoding for label classes
    one_hot_vector = np.zeros(num_classes, dtype=int)
    np.put(one_hot_vector, get_label(example), 1)
    return one_hot_vector


def separate_train_by_labels(train_data: np.ndarray, classes: collections.OrderedDict):
    """
    Separate training data by Bristol scale rating.

    Input:
        train_data: Array of training examples
        classes: Available classes that we are labelling

    Output:
        Training data separated by class (Bristol scale rating)
    """
    # Creates dict so we can store training data by label
    train_data_by_label = {key: list() for key in classes.values()}
    from itertools import zip_longest

    # Divides training data by label
    for label, num in zip_longest(classes.values(), classes.keys()):
        train_data_by_label[label] = train_data[np.where(train_data[:, 1] == str(num))]

    return train_data_by_label

def augment_image(img_path: str, save_to: str, img_num=200):
    """
    Augment a single image using a Keras Image Data Generator.

    Input:
        img_path: String containing image path
        save_to: Directory to save image
        img_num: Number of augmented images to produce based on that image

    Output:
        No output, but the augmented images will be saved to save_to
    """
    # Basic image data generator from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

    # Parameters for transformations
    datagen = ImageDataGenerator(
        rotation_range=360,  # degree range for random rotations
        width_shift_range=0.2,  # fraction of total width that we can shift the image
        height_shift_range=0.2,  # fraction of total height that we can shift the image
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
    )

    # Load the images for the Keras generator
    img = load_img(img_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    # Create the directory we will save to
    if os.path.isdir(save_to) == False:
        os.mkdir(save_to)

    # Augment the image img_num times and save as a .jpg
    i = 0
    for batch in datagen.flow(
        x,
        batch_size=1,
        save_to_dir=save_to,
        save_prefix="aug",
        save_format="jpg",
    ):
        i += 1
        if i > img_num:
            break


def augment_all_images(root_dir="training_data", new_dir="augmented_data"):
    """
    Recursively goes through the directory containing the images and augments all images.
    Saves images to the specified sub_dir.

    Input:
        root_dir: Starting directory containing subdirectories based on labels.
        Each subdirectory should then contain images.
        sub_dir: Directory to save augmented images

    Output:
        No output, but the augmented images will be saved to sub_dir in each class' directory
    """
    branch_dirs = []

    # Find all directories we will be working with
    for root_dir, dirs, files in os.walk(root_dir, topdown=False):
        for name in dirs:
            branch_dirs.append(name)

    # Create the new directory for augmented images as needed
    if os.path.isdir(new_dir) == False:
        os.mkdir(new_dir)

    # Augment all images using augment_image()
    for directory in branch_dirs:
        for filename in os.listdir(os.path.join(root_dir, directory)):
            # If the image is augmented somehow already, ignore it
            if "aug" in filename:
                continue
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                augment_image(
                    os.path.join(os.path.join(root_dir, directory), filename),
                    os.path.join(new_dir, directory),
                )
            else:
                # Removes all non-JPG, non-JPEG images
                os.remove(os.path.join(os.path.join(root_dir, directory), filename))

def downsample_train_data(train_data: np.ndarray, classes: collections.OrderedDict):
    """
    Downsamples the training data.
    """
    data = separate_train_by_labels(np.array(train_data), classes)
    train = []
    for label in data.keys():
        try:
            train = np.vstack((train, data[label][:9]))
        except:
            train = data[label][:9]
    shuffler = random.Random()
    shuffler.shuffle(train)
    return train

def crop_image_from_mask(args):
    """
    Crop an image based on the image mask.

    Input:
        args: A tuple of (2D numpy array of image data, 2D numpy array of mask data)

    Output:
        image: Numpy array of image with areas outside the mask cropped out.
    """
    # Parse args
    image = args[0]
    mask = args[1]
    
    # Get indices where mask is not 1
    indices = np.where(mask!=1)[:2]

    # Cut out data in image at indexes not containing mask
    image[indices] = [0, 0, 0]

    return image
