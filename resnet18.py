# Made with tutorial from: https://medium.com/@divyanshuraj.6815/learn-to-code-in-tensorflow2-part2-b1c448abbf1e

import tensorflow as tf


class CommonLayers(tf.keras.Model):
  def __init__(self, num_filters: int=0):
    super().__init__()
    self.num_filters = num_filters
    if num_filters != 0:
        self.conv = tf.keras.layers.Conv2D(
          filters = num_filters, kernel_size = 3, padding="SAME", kernel_initializer='glorot_uniform', use_bias=False
        )
    self.batch_norm = tf.keras.layers.BatchNormalization ()

  def call(self, inputs):
    if self.num_filters == 0:
      return tf.nn.relu (self.batch_norm (inputs))
    else:
      return tf.nn.relu (self.batch_norm (self.conv (inputs)))

class ResBlock(tf.keras.Model):
  def __init__ (self, c_out, flag_reshape, flag_common):
    super ().__init__ ()
    self.common1 = CommonLayers ()
    self.common2 = CommonLayers ()
    
    if flag_reshape == False:
      self.conv1 = tf.keras.layers.Conv2D(filters=c_out, kernel_size=3, padding="SAME", kernel_initializer='glorot_uniform', use_bias=False)
      self.conv2 = tf.keras.layers.Conv2D(filters=c_out, kernel_size=3, padding="SAME", kernel_initializer='glorot_uniform', use_bias=False)
    else:
      self.conv1 = tf.keras.layers.Conv2D(filters=c_out, kernel_size=3, padding="SAME", kernel_initializer='glorot_uniform', strides = 2, use_bias=False)
      self.conv2 = tf.keras.layers.Conv2D(filters=c_out, kernel_size=3, padding="SAME", kernel_initializer='glorot_uniform', use_bias=False)
      self.pool = tf.keras.layers.Conv2D(filters=c_out, kernel_size=1, strides = 2, kernel_initializer='glorot_uniform', use_bias=False)

    self.maxpool = tf.keras.layers.MaxPool2D ((4, 4))
    self.avgpool = tf.keras.layers.AveragePooling2D ((4, 4))
    self.flag_reshape = flag_reshape
    self.flag_common = flag_common
    self.c_out = c_out

  def call (self, inputs):
    h = self.conv2 (self.common1 (self.conv1 (inputs)))
    #if we want to reshape/reduce the image size or increase the number of channels
    if self.flag_reshape == True:
      h = h + self.pool (inputs)
    else:
      h = h + inputs
    
    #for last blocks
    if self.flag_common == False:
      return tf.keras.layers.concatenate ([self.maxpool (h), self.avgpool (h)])
    else:
      return self.common2 (h)

class ResNet18(tf.keras.Model):
  def __init__ (self, c = 64):
    super ().__init__ ()
    self.common = CommonLayers(c)
    self.blk1_1 = ResBlock(c, False, True)
    self.blk1_2 = ResBlock(c, False, True)
    self.blk2_1 = ResBlock(c * 2, True, True)
    self.blk2_2 = ResBlock(c * 2, False, True)
    self.blk3_1 = ResBlock(c * 4, True, True)
    self.blk3_2 = ResBlock(c * 4, False, True)
    self.blk4_1 = ResBlock(c * 8, True, True)
    self.blk4_2 = ResBlock(c * 8, False, False)
    
    self.linear = tf.keras.layers.Conv2D(filters = 10, kernel_size = 1, kernel_initializer='glorot_uniform', use_bias=False)
    self.flat = tf.keras.layers.Flatten()
    self.actn = tf.keras.layers.Activation ('softmax')
  
  def call (self, x, y):
    h = self.common(x)
    h = self.actn(self.flat (self.linear (self.blk4_2 (self.blk4_1 (self.blk3_2 (self.blk3_1 (self.blk2_2 (self.blk2_1 (self.blk1_2 (self.blk1_1 (h)))))))))))
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h, labels=y)
    loss = tf.reduce_sum(ce)
    correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(h, axis = 1), y), tf.float32))
    return loss, correct