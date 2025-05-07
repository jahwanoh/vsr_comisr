#!/usr/bin/env python3
# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Video Super-Resolution using COMISR.

This script applies COMISR super-resolution to an MP4 video file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import time

from absl import app
from absl import flags
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.io.gfile as gfile

# Handle tensorflow_addons compatibility issue
try:
    import tensorflow_addons as tfa
    has_tfa = True
except (ImportError, ModuleNotFoundError):
    print("Warning: tensorflow_addons import failed. Using fallback implementations.")
    has_tfa = False

from comisr.lib.model import fnet
from comisr.lib.model import generator_f
import comisr.lib.ops as ops


flags.DEFINE_string('input_video', None,
                    'Path to the input video file')
flags.DEFINE_string('output_video', None,
                    'Path to the output video file')
flags.DEFINE_string('checkpoint_path', None,
                    'If provided, the weight will be restored from the provided checkpoint')
flags.DEFINE_integer('num_resblock', 10,
                     'How many residual blocks are there in the generator')
flags.DEFINE_integer('vsr_scale', 4, 'Super resolution scale')
flags.DEFINE_boolean('use_ema', True, 'Use EMA variables')
flags.DEFINE_integer('start_frame', 0, 'Starting frame for processing')
flags.DEFINE_integer('end_frame', -1, 'Ending frame for processing (-1 for all frames)')
flags.DEFINE_float('fps', 0, 'Output video FPS (0 to use same as input)')

FLAGS = flags.FLAGS
tf.compat.v1.disable_eager_execution()


class VideoReader:
    """Helper class to read frames from a video file."""
    
    def __init__(self, video_path, start_frame=0, end_frame=-1):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.start_frame = start_frame
        self.end_frame = end_frame if end_frame > 0 else self.frame_count
        self.current_frame = 0
        
        # Seek to start frame
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.current_frame = start_frame
            
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.current_frame >= self.end_frame:
            raise StopIteration
            
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
            
        self.current_frame += 1
        # Convert from BGR to RGB
        return frame[:, :, ::-1]
        
    def __len__(self):
        return self.end_frame - self.start_frame
        
    def __del__(self):
        if self.cap is not None:
            self.cap.release()


def _get_ema_vars():
    """Gets all variables for which we maintain the moving average."""
    ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
    for v in tf.global_variables():
        # We maintain moving average not only for all trainable variables, but also
        # some other non-trainable variables including batch norm moving mean and
        # variance.
        if 'moving_mean' in v.name or 'moving_variance' in v.name:
            ema_vars.append(v)
    return list(set(ema_vars))


def gaussian_filter2d(image, sigma=1.5):
    """Fallback implementation of gaussian_filter2d if tensorflow_addons is not available."""
    # Create 1D kernels
    size = int(sigma * 4) * 2 + 1
    channels = image.get_shape().as_list()[-1]
    
    x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    kernel_1d = tf.exp(-tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, tf.float32), 2.0)))
    kernel_1d = kernel_1d / tf.reduce_sum(kernel_1d)
    
    # Expand to 2D kernel
    kernel_2d = tf.tensordot(kernel_1d, kernel_1d, axes=0)
    kernel_2d = tf.expand_dims(tf.expand_dims(kernel_2d, -1), -1)
    kernel_2d = tf.tile(kernel_2d, [1, 1, channels, 1])
    
    # Apply the filter
    padded_image = tf.pad(image, [[0, 0], [size//2, size//2], [size//2, size//2], [0, 0]], 'REFLECT')
    return tf.nn.depthwise_conv2d(padded_image, kernel_2d, strides=[1, 1, 1, 1], padding='VALID')


def dense_image_warp(image, flow):
    """Fallback implementation of dense_image_warp if tensorflow_addons is not available."""
    batch_size, height, width, channels = tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2], tf.shape(image)[3]
    
    # Create meshgrid
    grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
    grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), tf.float32)
    grid = tf.expand_dims(grid, axis=0)
    grid = tf.tile(grid, [batch_size, 1, 1, 1])
    
    # Add flow to grid
    pos = grid + flow
    
    # Clip to valid range
    pos = tf.clip_by_value(pos, 0, tf.cast([height-1, width-1], tf.float32))
    
    # Get integer and fractional parts of coordinates
    pos_floor = tf.floor(pos)
    pos_top_left = tf.cast(pos_floor, tf.int32)
    pos_bottom_right = tf.minimum(pos_top_left + 1, [height-1, width-1])
    pos_top_right = tf.stack([pos_top_left[..., 0], pos_bottom_right[..., 1]], axis=-1)
    pos_bottom_left = tf.stack([pos_bottom_right[..., 0], pos_top_left[..., 1]], axis=-1)
    
    # Calculate interpolation weights
    alpha = pos - pos_floor
    alpha_other = 1 - alpha
    
    # Gather pixel values and apply weights
    def gather_pixel(pos):
        y, x = pos[..., 0], pos[..., 1]
        indices = tf.stack([tf.range(batch_size, dtype=tf.int32)[:, None, None], y, x], axis=-1)
        return tf.gather_nd(image, indices)
    
    top_left = gather_pixel(pos_top_left)
    top_right = gather_pixel(pos_top_right)
    bottom_left = gather_pixel(pos_bottom_left)
    bottom_right = gather_pixel(pos_bottom_right)
    
    # Bilinear interpolation
    interp_top = alpha[..., 1:2] * top_right + alpha_other[..., 1:2] * top_left
    interp_bottom = alpha[..., 1:2] * bottom_right + alpha_other[..., 1:2] * bottom_left
    return alpha[..., 0:1] * interp_bottom + alpha_other[..., 0:1] * interp_top


def extract_detail_ops(image, sigma=1.5):
    """Extract details from the image tensors."""
    # input image is a 3D or 4D tensor with image range in [0, 1].
    if has_tfa:
        image_blur = tfa.image.gaussian_filter2d(image, sigma=sigma)
    else:
        image_blur = gaussian_filter2d(image, sigma=sigma)
    laplacian_image = (image - image_blur)
    return laplacian_image


def process_video(input_video, output_video, checkpoint_path, num_resblock, 
                 vsr_scale, use_ema, start_frame, end_frame, fps):
    """Process video with COMISR super-resolution."""
    
    if checkpoint_path is None:
        raise ValueError('The checkpoint file is needed to perform super-resolution.')
    
    # Initialize video reader
    video_reader = VideoReader(input_video, start_frame, end_frame)
    
    # Get the first frame to determine dimensions
    cap = cv2.VideoCapture(input_video)
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError(f"Could not read from video file: {input_video}")
    cap.release()
    
    # Convert BGR to RGB for processing
    first_frame = first_frame[:, :, ::-1]
    
    # Set input/output shapes
    input_shape = [1, first_frame.shape[0], first_frame.shape[1], 3]
    output_shape = [1, input_shape[1] * vsr_scale, input_shape[2] * vsr_scale, 3]
    oh = input_shape[1] - input_shape[1] // 8 * 8
    ow = input_shape[2] - input_shape[2] // 8 * 8
    paddings = tf.constant([[0, 0], [0, oh], [0, ow], [0, 0]])
    
    print('Input shape:', input_shape)
    print('Output shape:', output_shape)
    
    # Build the graph
    inputs_raw = tf.placeholder(tf.float32, shape=input_shape, name='inputs_raw')

    pre_inputs = tf.Variable(
        tf.zeros(input_shape), trainable=False, name='pre_inputs')
    pre_gen = tf.Variable(tf.zeros(output_shape), trainable=False, name='pre_gen')
    pre_warp = tf.Variable(
        tf.zeros(output_shape), trainable=False, name='pre_warp')

    transpose_pre = tf.space_to_depth(pre_warp, vsr_scale)
    inputs_all = tf.concat((inputs_raw, transpose_pre), axis=-1)
    with tf.variable_scope('generator'):
        gen_output = generator_f(
            inputs_all, 3, num_resblock, vsr_scale, reuse=False)
        # Deprocess the images outputed from the model, and assign things for next
        # frame
        with tf.control_dependencies([tf.assign(pre_inputs, inputs_raw)]):
            outputs = tf.assign(pre_gen, ops.deprocess(gen_output))

    inputs_frames = tf.concat((pre_inputs, inputs_raw), axis=-1)
    with tf.variable_scope('fnet'):
        gen_flow_lr = fnet(inputs_frames, reuse=False)
        gen_flow_lr = tf.pad(gen_flow_lr, paddings, 'SYMMETRIC')

        deconv_flow = gen_flow_lr
        deconv_flow = ops.conv2_tran(
            deconv_flow, 3, 64, 2, scope='deconv_flow_tran1')
        deconv_flow = tf.nn.relu(deconv_flow)
        deconv_flow = ops.conv2_tran(
            deconv_flow, 3, 64, 2, scope='deconv_flow_tran2')
        deconv_flow = tf.nn.relu(deconv_flow)
        deconv_flow = ops.conv2(deconv_flow, 3, 2, 1, scope='deconv_flow_conv')
        gen_flow = ops.upscale_x(gen_flow_lr * 4.0, scale=vsr_scale)
        gen_flow = deconv_flow + gen_flow

        gen_flow.set_shape(output_shape[:-1] + [2])
    
    # Use tensorflow_addons if available, otherwise use our fallback implementation
    if has_tfa:
        pre_warp_hi = tfa.image.dense_image_warp(pre_gen, gen_flow)
    else:
        pre_warp_hi = dense_image_warp(pre_gen, gen_flow)
    
    pre_warp_hi = pre_warp_hi + extract_detail_ops(pre_warp_hi)
    before_ops = tf.assign(pre_warp, pre_warp_hi)

    print('Finished building the network')

    if use_ema:
        moving_average_decay = 0.99
        global_step = tf.train.get_or_create_global_step()
        ema = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
        ema_vars = _get_ema_vars()

    # In inference time, we only need to restore the weight of the generator
    var_list = tf.trainable_variables()

    restore_vars_dict = {}
    if use_ema:
        for v in var_list:
            if re.match(v.name, '.*global_step.*'):
                restore_vars_dict[v.name[:-2]] = v
            else:
                restore_vars_dict[v.name[:-2] + '/ExponentialMovingAverage'] = v
    else:
        restore_vars_dict = var_list

    weight_initializer = tf.train.Saver(restore_vars_dict)

    # Define the initialization operation
    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_video)
    if output_dir and not gfile.exists(output_dir):
        gfile.makedirs(output_dir)
    
    # Setup video writer
    output_fps = video_reader.fps if fps == 0 else fps
    output_width = output_shape[2]
    output_height = output_shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create a temporary file path for the video
    temp_output_video = output_video + '.temp.mp4'
    writer = cv2.VideoWriter(
        temp_output_video, fourcc, output_fps, (output_width, output_height))
    
    if not writer.isOpened():
        raise ValueError(f"Could not open video writer for: {output_video}")

    with tf.Session(config=config) as sess:
        # Load the pretrained model
        sess.run(init_op)
        sess.run(local_init_op)

        print('Loading weights from checkpoint')
        weight_initializer.restore(sess, checkpoint_path)
        
        total_frames = len(video_reader)
        total_time = 0
        
        print(f'Processing {total_frames} frames...')
        
        # Process each frame
        for i, frame in enumerate(video_reader):
            # Normalize input frame to [0, 1]
            input_frame = np.array([frame]).astype(np.float32) / 255.0
            
            feed_dict = {inputs_raw: input_frame}
            t0 = time.time()
            
            if i != 0:
                sess.run(before_ops, feed_dict=feed_dict)
            
            output_frame = sess.run(outputs, feed_dict=feed_dict)
            frame_time = time.time() - t0
            total_time += frame_time
            
            if i >= 5:  # Skip first 5 frames as warmup
                # Convert output to BGR for OpenCV
                output_img = np.clip(output_frame[0] * 255.0, 0, 255).astype(np.uint8)
                output_img = output_img[:, :, ::-1]  # RGB to BGR
                writer.write(output_img)
                
                if (i + 1) % 10 == 0:
                    print(f'Processed frame {i+1}/{total_frames}, '
                          f'time: {frame_time:.3f}s, '
                          f'average: {total_time/(i+1):.3f}s/frame')
            else:
                print(f'Warming up, frame {i+1}/5')
    
    writer.release()
    
    # Create a final version with proper encoding
    print(f"Re-encoding final video to {output_video}")
    os.system(f"ffmpeg -i {temp_output_video} -c:v libx264 -crf 18 -preset slow {output_video} -y")
    
    # Clean up temporary file
    if os.path.exists(temp_output_video):
        os.remove(temp_output_video)
    
    print(f'Completed video processing. Output saved to: {output_video}')
    print(f'Total processing time: {total_time:.2f}s, '
          f'Average: {total_time/total_frames:.3f}s/frame')


def main(argv):
    del argv  # Unused
    
    # Check required flags
    if not FLAGS.input_video:
        raise ValueError("--input_video is required")
    if not FLAGS.output_video:
        raise ValueError("--output_video is required")
    if not FLAGS.checkpoint_path:
        raise ValueError("--checkpoint_path is required")
    
    process_video(
        input_video=FLAGS.input_video,
        output_video=FLAGS.output_video,
        checkpoint_path=FLAGS.checkpoint_path,
        num_resblock=FLAGS.num_resblock,
        vsr_scale=FLAGS.vsr_scale,
        use_ema=FLAGS.use_ema,
        start_frame=FLAGS.start_frame,
        end_frame=FLAGS.end_frame,
        fps=FLAGS.fps
    )


if __name__ == '__main__':
    app.run(main) 