import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
import numpy as np

def get_classifier(configs: dict, num_classes: int):
    """
    Set-up and compile a CNN for transfer learning based on configs

    Input:
        configs: Dictionary of configurations for the classifier
        num_classes: int number of classes in data being trained on

    Output:
        Compiled tensorflow keras classifier
    """
    # Get model string
    model_str = configs['model'].lower()
    if model_str == 'resnet50':
        return get_resnet50(configs, num_classes)
    
    return get_mobile_net_v2(configs, num_classes) 


def get_mobile_net_v2(configs: dict, num_classes: int):
    """
    Set-up and compile a Mobile Net v2 CNN with some dense layers for transfer learning.

    Input:
        configs: Dictionary of configurations for the classifier
        num_classes: int number of classes in data being trained on

    Output:
        Compiled tensorflow keras classifier
    """
    # Set up model details
    image_size = tuple(configs['image_dimensions'])
    module_selection = ("mobilenet_v2_100_" + str(image_size[0]), image_size[0])
    handle_base, pixels = module_selection
    model_link = (
        "https://tfhub.dev/google/imagenet/mobilenet_v2_100_" + str(image_size[0]) + "/feature_vector/4"
    )
    print("Using {} with input size {}".format(model_link, image_size))

    # Get the Mobile Net v2
    print("Building model with", model_link)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=image_size + (3,)),
            hub.KerasLayer(model_link, trainable=False),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(
                num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.0001)
            ),
        ]
    )
    model.build((None,) + image_size + (3,))
    model.summary()

    # Compile the classifier
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=configs["learning_rate"]),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    )

    return model

def get_resnet50(configs: dict, num_classes: int):
    """
    Set-up and compile a ResNet50 CNN with some dense layers for transfer learning.

    Input:
        configs: Dictionary of configurations for the classifier
        num_classes: int number of classes in data being trained on

    Output:
        Compiled tensorflow keras classifier
    """

    # Get the ResNet
    image_size = tuple(configs['image_dimensions'])
    resnet = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=(224, 224, 3),
    )
    resnet.trainable = False
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=image_size + (3,)),
            resnet,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(
                num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.0001)
            ),
        ]
    )
    model.build((None,) + image_size + (3,))
    model.summary()

    # Compile the classifier
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=configs["learning_rate"]),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    )

    return model

# https://github.com/Pepslee/tensorflow-contrib/blob/master/unpooling.py
class MaxUnpool2D(tf.keras.layers.Layer):
  def __init__(self, stride=2, batch_size=32, name='maxunpool'):
    super(MaxUnpool2D, self).__init__(name=name)
    self.stride = stride
    self.batch_size = batch_size

  def build(self, input_shape):
    pass

  def call(self, input, mask):
    x = input

    input_shape = input.get_shape().as_list()
    
    # If compiling network, use batch_size
    if input_shape[0] is None:
        input_shape[0] = self.batch_size

    strides = [1, self.stride, self.stride, 1]
    output_shape = (input_shape[0],
                    input_shape[1] * strides[1],
                    input_shape[2] * strides[2],
                    input_shape[3])


    flat_output_shape = [output_shape[0], np.prod(output_shape[1:])]
    with tf.name_scope(self.name):
        flat_input_size = tf.size(x)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=mask.dtype),
                                 shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(mask) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        mask_ = tf.reshape(mask, [flat_input_size, 1])
        mask_ = tf.concat([b, mask_], 1)

        x_ = tf.reshape(x, [flat_input_size])
        ret = tf.scatter_nd(mask_, x_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
        return ret

def get_segnet(configs: dict, num_classes: int=1):
    """
    Set-up and compile a SegNet for image segmentation

    Input:
        configs: Dictionary of configurations for the classifier
        num_classes: int number of classes in data being trained on

    Output:
        Compiled tensorflow SegNet
    """
    # Get parameters
    kernel=3
    input_shape = tuple(configs['image_dimensions']) + (3,)
    batch_size = configs['batch_size']
        
    print('Building SegNet')

    # encoder
    inputs = tf.keras.layers.Input(shape=input_shape)
    conv_1 = tf.keras.layers.Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_1 = tf.keras.layers.Activation("relu")(conv_1)
    conv_2 = tf.keras.layers.Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    conv_2 = tf.keras.layers.Activation("relu")(conv_2)

    pool_1, mask_1 = tf.nn.max_pool_with_argmax(conv_2, ksize=[1, 2, 2, 1], 
                                                strides=[1, 2, 2, 1], padding='SAME')

    conv_3 = tf.keras.layers.Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    conv_3 = tf.keras.layers.Activation("relu")(conv_3)
    conv_4 = tf.keras.layers.Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
    conv_4 = tf.keras.layers.Activation("relu")(conv_4)

    pool_2, mask_2 = tf.nn.max_pool_with_argmax(conv_4, ksize=[1, 2, 2, 1], 
                                                strides=[1, 2, 2, 1], padding='SAME')

    conv_5 = tf.keras.layers.Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = tf.keras.layers.BatchNormalization()(conv_5)
    conv_5 = tf.keras.layers.Activation("relu")(conv_5)
    conv_6 = tf.keras.layers.Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = tf.keras.layers.BatchNormalization()(conv_6)
    conv_6 = tf.keras.layers.Activation("relu")(conv_6)
    conv_7 = tf.keras.layers.Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = tf.keras.layers.BatchNormalization()(conv_7)
    conv_7 = tf.keras.layers.Activation("relu")(conv_7)

    pool_3, mask_3 = tf.nn.max_pool_with_argmax(conv_7, ksize=[1, 2, 2, 1], 
                                                strides=[1, 2, 2, 1], padding='SAME')

    conv_8 = tf.keras.layers.Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = tf.keras.layers.BatchNormalization()(conv_8)
    conv_8 = tf.keras.layers.Activation("relu")(conv_8)
    conv_9 = tf.keras.layers.Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = tf.keras.layers.BatchNormalization()(conv_9)
    conv_9 = tf.keras.layers.Activation("relu")(conv_9)
    conv_10 = tf.keras.layers.Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = tf.keras.layers.BatchNormalization()(conv_10)
    conv_10 = tf.keras.layers.Activation("relu")(conv_10)

    pool_4, mask_4 = tf.nn.max_pool_with_argmax(conv_10, ksize=[1, 2, 2, 1], 
                                                strides=[1, 2, 2, 1], padding='SAME')

    conv_11 = tf.keras.layers.Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = tf.keras.layers.BatchNormalization()(conv_11)
    conv_11 = tf.keras.layers.Activation("relu")(conv_11)
    conv_12 = tf.keras.layers.Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = tf.keras.layers.BatchNormalization()(conv_12)
    conv_12 = tf.keras.layers.Activation("relu")(conv_12)
    conv_13 = tf.keras.layers.Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = tf.keras.layers.BatchNormalization()(conv_13)
    conv_13 = tf.keras.layers.Activation("relu")(conv_13)

    pool_5, mask_5 = tf.nn.max_pool_with_argmax(conv_13, ksize=[1, 2, 2, 1], 
                                                strides=[1, 2, 2, 1], padding='SAME')
    print("Build encoder done..")

    # decoder

    unpool_1 = MaxUnpool2D(batch_size=batch_size, name='maxunpool1')(pool_5, mask_5)

    conv_14 = tf.keras.layers.Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = tf.keras.layers.BatchNormalization()(conv_14)
    conv_14 = tf.keras.layers.Activation("relu")(conv_14)
    conv_15 = tf.keras.layers.Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = tf.keras.layers.BatchNormalization()(conv_15)
    conv_15 = tf.keras.layers.Activation("relu")(conv_15)
    conv_16 = tf.keras.layers.Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = tf.keras.layers.BatchNormalization()(conv_16)
    conv_16 = tf.keras.layers.Activation("relu")(conv_16)

    unpool_2 = MaxUnpool2D(batch_size=batch_size, name='maxunpool2')(conv_16, mask_4)

    conv_17 = tf.keras.layers.Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = tf.keras.layers.BatchNormalization()(conv_17)
    conv_17 = tf.keras.layers.Activation("relu")(conv_17)
    conv_18 = tf.keras.layers.Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = tf.keras.layers.BatchNormalization()(conv_18)
    conv_18 = tf.keras.layers.Activation("relu")(conv_18)
    conv_19 = tf.keras.layers.Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = tf.keras.layers.BatchNormalization()(conv_19)
    conv_19 = tf.keras.layers.Activation("relu")(conv_19)

    unpool_3 = MaxUnpool2D(batch_size=batch_size, name='maxunpool3')(conv_19, mask_3)

    conv_20 = tf.keras.layers.Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = tf.keras.layers.BatchNormalization()(conv_20)
    conv_20 = tf.keras.layers.Activation("relu")(conv_20)
    conv_21 = tf.keras.layers.Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = tf.keras.layers.BatchNormalization()(conv_21)
    conv_21 = tf.keras.layers.Activation("relu")(conv_21)
    conv_22 = tf.keras.layers.Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = tf.keras.layers.BatchNormalization()(conv_22)
    conv_22 = tf.keras.layers.Activation("relu")(conv_22)

    unpool_4 = MaxUnpool2D(batch_size=batch_size, name='maxunpool4')(conv_22, mask_2)

    conv_23 = tf.keras.layers.Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = tf.keras.layers.BatchNormalization()(conv_23)
    conv_23 = tf.keras.layers.Activation("relu")(conv_23)
    conv_24 = tf.keras.layers.Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = tf.keras.layers.BatchNormalization()(conv_24)
    conv_24 = tf.keras.layers.Activation("relu")(conv_24)

    unpool_5 = MaxUnpool2D(batch_size=batch_size, name='maxunpool5')(conv_24, mask_1)

    conv_25 = tf.keras.layers.Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = tf.keras.layers.BatchNormalization()(conv_25)
    conv_25 = tf.keras.layers.Activation("relu")(conv_25)

    conv_26 = tf.keras.layers.Convolution2D(num_classes, (1, 1), padding="valid")(conv_25)
    conv_26 = tf.keras.layers.BatchNormalization()(conv_26)
    # conv_26 = tf.keras.layers.Reshape(
    #         (input_shape[0]*input_shape[1], num_classes),
    #         input_shape=(input_shape[0], input_shape[1], num_classes))(conv_26)

    outputs = tf.keras.layers.Activation("sigmoid")(conv_26)
    print("Build decoder done..")

    # Compile the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SegNet")
    model.compile(tfa.optimizers.SGDW(learning_rate=0.1, momentum=0.9, weight_decay=0.05), tf.keras.losses.BinaryCrossentropy())

    # Set weights in encoder layers for transfer learning from VGG16 weights
    vgg16 = tf.keras.applications.VGG16(include_top=False, input_tensor=inputs)
    j = 0
    for i in range(len(vgg16.layers)):
        copied = False
        while (not copied):
            try:
                model.layers[j].set_weights(vgg16.layers[i].get_weights())
                model.layers[j].trainable = False
                copied = True
            except:
                j += 1
        j += 1

    # Print out model summary
    model.summary()

    return model