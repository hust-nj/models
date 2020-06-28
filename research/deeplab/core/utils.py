# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This script contains utility functions."""
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import slim as contrib_slim
import math


slim = contrib_slim


# Quantized version of sigmoid function.
q_sigmoid = lambda x: tf.nn.relu6(x + 3) * 0.16667


def resize_bilinear(images, size, output_dtype=tf.float32):
  """Returns resized images as output_type.

  Args:
    images: A tensor of size [batch, height_in, width_in, channels].
    size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new size
      for the images.
    output_dtype: The destination type.
  Returns:
    A tensor of size [batch, height_out, width_out, channels] as a dtype of
      output_dtype.
  """
  images = tf.image.resize_bilinear(images, size, align_corners=True)
  return tf.cast(images, dtype=output_dtype)


def scale_dimension(dim, scale):
  """Scales the input dimension.

  Args:
    dim: Input dimension (a scalar or a scalar Tensor).
    scale: The amount of scaling applied to the input.

  Returns:
    Scaled dimension.
  """
  if isinstance(dim, tf.Tensor):
    return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0, dtype=tf.int32)
  else:
    return int((float(dim) - 1.0) * scale + 1.0)


def split_separable_conv2d(inputs,
                           filters,
                           kernel_size=3,
                           rate=1,
                           weight_decay=0.00004,
                           depthwise_weights_initializer_stddev=0.33,
                           pointwise_weights_initializer_stddev=0.06,
                           scope=None):
  """Splits a separable conv2d into depthwise and pointwise conv2d.

  This operation differs from `tf.layers.separable_conv2d` as this operation
  applies activation function between depthwise and pointwise conv2d.

  Args:
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    depthwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for depthwise convolution.
    pointwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for pointwise convolution.
    scope: Optional scope for the operation.

  Returns:
    Computed features after split separable conv2d.
  """
  outputs = slim.separable_conv2d(
      inputs,
      None,
      kernel_size=kernel_size,
      depth_multiplier=1,
      rate=rate,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=depthwise_weights_initializer_stddev),
      weights_regularizer=None,
      scope=scope + '_depthwise')
  return slim.conv2d(
      outputs,
      filters,
      1,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=pointwise_weights_initializer_stddev),
      weights_regularizer=slim.l2_regularizer(weight_decay),
      scope=scope + '_pointwise')


def get_label_weight_mask(labels, ignore_label, num_classes, label_weights=1.0):
  """Gets the label weight mask.

  Args:
    labels: A Tensor of labels with the shape of [-1].
    ignore_label: Integer, label to ignore.
    num_classes: Integer, the number of semantic classes.
    label_weights: A float or a list of weights. If it is a float, it means all
      the labels have the same weight. If it is a list of weights, then each
      element in the list represents the weight for the label of its index, for
      example, label_weights = [0.1, 0.5] means the weight for label 0 is 0.1
      and the weight for label 1 is 0.5.

  Returns:
    A Tensor of label weights with the same shape of labels, each element is the
      weight for the label with the same index in labels and the element is 0.0
      if the label is to ignore.

  Raises:
    ValueError: If label_weights is neither a float nor a list, or if
      label_weights is a list and its length is not equal to num_classes.
  """
  if not isinstance(label_weights, (float, list)):
    raise ValueError(
        'The type of label_weights is invalid, it must be a float or a list.')

  if isinstance(label_weights, list) and len(label_weights) != num_classes:
    raise ValueError(
        'Length of label_weights must be equal to num_classes if it is a list, '
        'label_weights: %s, num_classes: %d.' % (label_weights, num_classes))

  not_ignore_mask = tf.not_equal(labels, ignore_label)
  not_ignore_mask = tf.cast(not_ignore_mask, tf.float32)
  if isinstance(label_weights, float):
    return not_ignore_mask * label_weights

  label_weights = tf.constant(label_weights, tf.float32)
  weight_mask = tf.einsum('...y,y->...',
                          tf.one_hot(labels, num_classes, dtype=tf.float32),
                          label_weights)
  return tf.multiply(not_ignore_mask, weight_mask)


def get_batch_norm_fn(sync_batch_norm_method):
  """Gets batch norm function.

  Currently we only support the following methods:
    - `None` (no sync batch norm). We use slim.batch_norm in this case.

  Args:
    sync_batch_norm_method: String, method used to sync batch norm.

  Returns:
    Batchnorm function.

  Raises:
    ValueError: If sync_batch_norm_method is not supported.
  """
  if sync_batch_norm_method == 'None':
    return slim.batch_norm
  else:
    raise ValueError('Unsupported sync_batch_norm_method.')


def get_batch_norm_params(decay=0.9997,
                          epsilon=1e-5,
                          center=True,
                          scale=True,
                          is_training=True,
                          sync_batch_norm_method='None',
                          initialize_gamma_as_zeros=False):
  """Gets batch norm parameters.

  Args:
    decay: Float, decay for the moving average.
    epsilon: Float, value added to variance to avoid dividing by zero.
    center: Boolean. If True, add offset of `beta` to normalized tensor. If
      False,`beta` is ignored.
    scale: Boolean. If True, multiply by `gamma`. If False, `gamma` is not used.
    is_training: Boolean, whether or not the layer is in training mode.
    sync_batch_norm_method: String, method used to sync batch norm.
    initialize_gamma_as_zeros: Boolean, initializing `gamma` as zeros or not.

  Returns:
    A dictionary for batchnorm parameters.

  Raises:
    ValueError: If sync_batch_norm_method is not supported.
  """
  batch_norm_params = {
      'is_training': is_training,
      'decay': decay,
      'epsilon': epsilon,
      'scale': scale,
      'center': center,
  }
  if initialize_gamma_as_zeros:
    if sync_batch_norm_method == 'None':
      # Slim-type gamma_initialier.
      batch_norm_params['param_initializers'] = {
          'gamma': tf.zeros_initializer(),
      }
    else:
      raise ValueError('Unsupported sync_batch_norm_method.')
  return batch_norm_params



def nonLocal2d_nowd(inputs,
                    filters,
                    downsample=True,
                    lr_mult=None,
                    use_out=False,
                    out_bn=False,
                    whiten_type=['in_nostd'],
                    weight_init_scale=1.0,
                    with_gc=False,
                    with_nl=True,
                    eps=1e-5,
                    nowd=['nl'],
                    scope=None):
  residual = inputs
  if downsample:
    inputs = slim.max_pool2d(inputs, 2, stride=2)
  
  inputs_shape = inputs.get_shape().as_list()
  if use_out:
    # N,H',W',C'
    value = slim.conv2d(inputs, filters, 1)
  else:
    # N,H',W',C'
    value = slim.conv2d(inputs, inputs_shape[3], 1, biases_initializer=None)
  
  value_shape = value.get_shape().as_list()
  # N, H'xW', C'
  value = tf.reshape(value, (-1, value_shape[1]*value_shape[2], value_shape[3]))

  out_sim = None

  if with_nl:
    # N,H,W,C
    query = slim.conv2d(residual, filters, 1)
    query_shape = query.get_shape().as_list()
    # N,HxW,C
    query = tf.reshape(query, (-1, query_shape[1]*query_shape[2], query_shape[3]))

    # N,H',W',C
    key = slim.conv2d(inputs, filters, 1)
    key_shape = key.get_shape().as_list()
    # N,H'xW',C
    key = tf.reshape(key, (-1, key_shape[1]*key_shape[2], key_shape[3]))

    if 'in_nostd' in whiten_type:
        key_mean = tf.reduce_mean(key, axis=1, keepdims=True)
        query_mean = tf.reduce_mean(query, axis=1, keepdims=True)
        key -= key_mean
        query -= query_mean
    elif 'in' in whiten_type:
        key_mean, key_var = tf.nn.moments(key, 1, keepdims=True)
        query_mean, query_var = tf.nn.moments(query, 1, keepdims=True)
        key -= key_mean
        query -= query_mean
        key = key / torch.sqrt(key_var + eps)
        query = query / torch.sqrt(query_var + eps)
    elif 'ln_nostd' in whiten_type:
        key_mean = tf.reduce_mean(key, [1,2], keepdims=True)
        query_mean = tf.reduce_mean(query, [1,2], keepdims=True)
        key -= key_mean
        query -= query_mean
    elif 'ln' in whiten_type:
        key_mean, key_var = tf.nn.moments(key, [1,2], keepdims=True)
        query_mean, query_var = tf.nn.moments(query, [1,2], keepdims=True)
        key -= key_mean
        query -= query_mean
        key = key / torch.sqrt(key_var + eps)
        query = query / torch.sqrt(query_var + eps)
    elif 'fln_nostd' in whiten_type :
        key_mean = tf.reduce_mean(key, [0,1,2], keepdims=True)
        query_mean = tf.reduce_mean(query, [0,1,2], keepdims=True)
        key -= key_mean
        query -= query_mean
    elif 'fln' in whiten_type:
        key_mean, key_var = tf.nn.moments(key, [0,1,2], keepdims=True)
        query_mean, query_var = tf.nn.moments(query, [0,1,2], keepdims=True)
        key -= key_mean
        query -= query_mean
        key = key / torch.sqrt(key_var + eps)
        query = query / torch.sqrt(query_var + eps)

    # N,HxW,H'xW'
    sim_map = tf.matmul(query, key, transpose_b=True)
    ### cancel temp and scale
    scale = math.sqrt(filters)
    if 'nl' not in nowd:
        sim_map = sim_map/scale
    sim_map = tf.nn.softmax(sim_map, axis=2)

    # [N, H x W, C']
    out_sim = tf.matmul(sim_map, value)
    # [N, H, W, C']
    out_sim = tf.reshape(out_sim, (-1, query_shape[1], query_shape[2], inputs_shape[-1]))
    gamma = slim.model_variable('gamma', shape=[], initializer=tf.zeros_initializer())
    out_sim = gamma * out_sim

  if with_gc:
    # N, H', W', 1
    mask = slim.conv2d(inputs, 1, 1)
    mask_shape = mask.get_shape().as_list()
    # N, H'xW', 1
    mask = tf.reshape(mask, (-1, mask_shape[1]*mask_shape[2], 1))
    mask = tf.nn.softmax(mask, axis=1)

    # N, 1, 1, C'
    out_gc = tf.expand_dims(tf.matmul(mask, value, transpose_a=True),1)

    if out_sim is not None:
      out_sim = out_sim + out_gc
    else:
      out_sim = out_gc

  if use_out:
    out_sim = slim.conv2d(out_sim, inputs_shape[3], 1, biases_initializer=None)
  
  if out_bn:
    out_sim = slim.batch_norm(out_sim)
  
  out = out_sim + residual
    
  return out
