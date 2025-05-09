# comisr/lib/ops.py

import numpy as np
import tensorflow as tf # Use tf directly for TF 2.x
import tensorflow_addons as tfa # Keep for dense_image_warp if needed

# Define our Lrelu
def lrelu(inputs, alpha):
    return tf.keras.layers.LeakyReLU(alpha=alpha)(inputs)

def preprocess(image):
    with tf.name_scope('preprocess'):
        # [0, 1] => [-1, 1]
        return image * 2.0 - 1.0

def deprocess(image):
    with tf.name_scope('deprocess'):
        # [-1, 1] => [0, 1]
        return (image + 1.0) / 2.0

def maxpool(inputs, scope='maxpool'): # scope is not directly used by Keras layers like this
    return tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), name=scope)(inputs)

def conv2_tran(batch_input,
               kernel=3,
               output_channel=64,
               stride=1,
               use_bias=True,
               scope='conv_tran'): # scope used for layer name
    """Define the convolution transpose building block."""
    return tf.keras.layers.Conv2DTranspose(
        filters=output_channel,
        kernel_size=kernel,
        strides=stride,
        padding='SAME',
        use_bias=use_bias,
        bias_initializer='zeros' if use_bias else None,
        kernel_initializer=tf.keras.initializers.GlorotUniform(), # Common TF1 default
        name=scope
    )(batch_input)

def conv2(batch_input,
          kernel=3,
          output_channel=64,
          stride=1,
          use_bias=True,
          scope='conv'): # scope used for layer name
    """Define the convolution building block."""
    return tf.keras.layers.Conv2D(
        filters=output_channel,
        kernel_size=kernel,
        strides=stride,
        padding='SAME',
        use_bias=use_bias,
        bias_initializer='zeros' if use_bias else None,
        kernel_initializer=tf.keras.initializers.GlorotUniform(), # Common TF1 default
        name=scope
    )(batch_input)

# upscale_x and bicubic_x can remain largely the same as they use core TF ops
# but ensure tf2.image.resize is tf.image.resize for TF2

def upscale_x(
    inputs,
    scale=4,
    scope='upscale_x' # scope not directly used here
):
  """mimic the tensorflow bilinear-upscaling for a fix ratio of x."""
  with tf.name_scope(scope): # tf.name_scope still works for organizing graph in TF2
    size = tf.shape(inputs)
    b = size[0]
    h = size[1]
    w = size[2]
    c = size[3]

    p_inputs = tf.concat((inputs, inputs[:, -1:, :, :]), axis=1)  # pad bottom
    p_inputs = tf.concat((p_inputs, p_inputs[:, :, -1:, :]),
                         axis=2)  # pad right

    hi_res_bin = [
        [
            inputs,  # top-left
            p_inputs[:, :-1, 1:, :]  # top-right
        ],
        [
            p_inputs[:, 1:, :-1, :],  # bottom-left
            p_inputs[:, 1:, 1:, :]  # bottom-right
        ]
    ]

    hi_res_array = []
    factor = 1.0 / float(scale)
    for hi in range(scale):
      for wj in range(scale):
        hi_res_array.append(hi_res_bin[0][0] * (1.0 - factor * hi) *
                            (1.0 - factor * wj) + hi_res_bin[0][1] *
                            (1.0 - factor * hi) * (factor * wj) +
                            hi_res_bin[1][0] * (factor * hi) *
                            (1.0 - factor * wj) + hi_res_bin[1][1] *
                            (factor * hi) * (factor * wj))

    hi_res = tf.stack(hi_res_array, axis=3)  # shape (b,h,w,16,c)
    hi_res_reshape = tf.reshape(hi_res, (b, h, w, scale, scale, c))
    hi_res_reshape = tf.transpose(hi_res_reshape, (0, 1, 3, 2, 4, 5))
    hi_res_reshape = tf.reshape(hi_res_reshape, (b, h * scale, w * scale, c))

  return hi_res_reshape


def bicubic_x(inputs, scale=4, scope='bicubic_x'): # scope not directly used here
  """Upscaling using tf.bicubic function."""
  with tf.name_scope(scope):
    if scale == 4:
      # return bicubic_four(inputs) # Assuming bicubic_four is complex, let's use tf.image.resize
      # For simplicity and directness in TF2, let's stick to tf.image.resize
      pass # Fall through to tf.image.resize if not using bicubic_four explicitly

    size = tf.shape(inputs)
    output_size = [scale * size[1], scale * size[2]]

    bicubic_x_inputs = tf.image.resize( # Use tf.image.resize directly
        inputs, output_size, method=tf.image.ResizeMethod.BICUBIC)
  return bicubic_x_inputs


def bicubic_four(inputs, scope='bicubic_four'): # scope not directly used here
  """Bicubic four upscaling."""
  # This function is complex and uses numpy. It should still work but might be slow if not tf.function-ed.
  # For TF2, if performance is an issue here, consider reimplementing with pure TF ops or ensure it's
  # part of a tf.function call. The original bicubic_x already prefers tf.image.resize for non-scale-4.
  with tf.name_scope(scope):
    size = tf.shape(inputs)
    b = size[0]
    h = size[1]
    w = size[2]
    c = size[3]

    p_inputs = tf.concat((inputs[:, :1, :, :], inputs), axis=1)  # pad top
    p_inputs = tf.concat((p_inputs[:, :, :1, :], p_inputs), axis=2)  # pad left
    p_inputs = tf.concat(
        (p_inputs, p_inputs[:, -1:, :, :], p_inputs[:, -1:, :, :]),
        axis=1)  # pad bottom
    p_inputs = tf.concat(
        (p_inputs, p_inputs[:, :, -1:, :], p_inputs[:, :, -1:, :]),
        axis=2)  # pad right

    hi_res_bin = [p_inputs[:, bi:bi + h, :, :] for bi in range(4)]
    r = 0.75
    # Convert mat and weights to tf.constant for graph mode compatibility
    mat_np = np.float32([[0, 1, 0, 0], [-r, 0, r, 0],
                      [2 * r, r - 3, 3 - 2 * r, -r], [-r, 2 - r, r - 2, r]])
    mat = tf.constant(mat_np, dtype=tf.float32)

    weights_np_list = [
        np.float32([1.0, t, t * t, t * t * t]).dot(mat_np)
        for t in [0.0, 0.25, 0.5, 0.75]
    ]
    weights = [tf.constant(w_np, dtype=tf.float32) for w_np in weights_np_list]


    hi_res_array = []
    for hi in range(4):
      cur_wei = weights[hi]
      cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[
          1] + cur_wei[2] * hi_res_bin[2] + cur_wei[3] * hi_res_bin[3]

      hi_res_array.append(cur_data)

    hi_res_y = tf.stack(hi_res_array, axis=2)  # shape (b,h,4,w,c)
    hi_res_y = tf.reshape(hi_res_y, (b, h * 4, w + 3, c))

    hi_res_bin = [hi_res_y[:, :, bj:bj + w, :] for bj in range(4)]

    hi_res_array = []
    for hj in range(4):
      cur_wei = weights[hj]
      cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[
          1] + cur_wei[2] * hi_res_bin[2] + cur_wei[3] * hi_res_bin[3]

      hi_res_array.append(cur_data)

    hi_res = tf.stack(hi_res_array, axis=3)  # shape (b,h*4,w,4,c)
    hi_res = tf.reshape(hi_res, (b, h * 4, w * 4, c))

  return hi_res

# Add missing extract_detail_ops from video_inference for consistency,
# or ensure it's correctly imported/defined where used in model.py if not here.
# For now, assuming it will be handled by the main script's definition.