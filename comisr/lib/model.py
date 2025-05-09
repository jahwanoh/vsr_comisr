# comisr/lib/model.py

import tensorflow as tf
# Ensure ops is imported from the refactored version
from . import ops # if in the same package, or from comisr.lib import ops

class FNet(tf.keras.Model):
    def __init__(self, name='fnet', **kwargs):
        super(FNet, self).__init__(name=name, **kwargs)
        # Define layers for down_block
        self.enc_conv1_1 = ops.conv2
        self.enc_lrelu1_1 = ops.lrelu
        self.enc_conv1_2 = ops.conv2
        self.enc_lrelu1_2 = ops.lrelu
        self.enc_pool1 = ops.maxpool

        self.enc_conv2_1 = ops.conv2
        self.enc_lrelu2_1 = ops.lrelu
        self.enc_conv2_2 = ops.conv2
        self.enc_lrelu2_2 = ops.lrelu
        self.enc_pool2 = ops.maxpool

        self.enc_conv3_1 = ops.conv2
        self.enc_lrelu3_1 = ops.lrelu
        self.enc_conv3_2 = ops.conv2
        self.enc_lrelu3_2 = ops.lrelu
        self.enc_pool3 = ops.maxpool

        # Define layers for up_block
        self.dec_conv1_1 = ops.conv2
        self.dec_lrelu1_1 = ops.lrelu
        self.dec_conv1_2 = ops.conv2
        self.dec_lrelu1_2 = ops.lrelu
        # self.dec_resize1 # resize handled in call

        self.dec_conv2_1 = ops.conv2
        self.dec_lrelu2_1 = ops.lrelu
        self.dec_conv2_2 = ops.conv2
        self.dec_lrelu2_2 = ops.lrelu
        # self.dec_resize2

        self.dec_conv3_1 = ops.conv2
        self.dec_lrelu3_1 = ops.lrelu
        self.dec_conv3_2 = ops.conv2
        self.dec_lrelu3_2 = ops.lrelu
        # self.dec_resize3

        # Output stage
        self.out_conv1 = ops.conv2
        self.out_lrelu1 = ops.lrelu
        self.out_conv2 = ops.conv2


    def _down_block(self, inputs, output_channel, conv1_op, lrelu1_op, conv2_op, lrelu2_op, pool_op, scope_suffix):
        # In Keras models, scope is handled by layer names or model name prefix
        net = conv1_op(inputs, 3, output_channel, stride=1, use_bias=True, scope=f'conv_1_{scope_suffix}')
        net = lrelu1_op(net, 0.2)
        net = conv2_op(net, 3, output_channel, stride=1, use_bias=True, scope=f'conv_2_{scope_suffix}') # Original used stride=1 here
        net = lrelu2_op(net, 0.2)
        net = pool_op(net, scope=f'pool_{scope_suffix}')
        return net

    def _up_block(self, inputs, output_channel, conv1_op, lrelu1_op, conv2_op, lrelu2_op, scope_suffix):
        net = conv1_op(inputs, 3, output_channel, stride=1, use_bias=True, scope=f'conv_1_{scope_suffix}')
        net = lrelu1_op(net, 0.2)
        net = conv2_op(net, 3, output_channel, stride=1, use_bias=True, scope=f'conv_2_{scope_suffix}')
        net = lrelu2_op(net, 0.2)
        
        # Get dynamic shape for resizing
        new_shape = tf.shape(net)[1:3] * 2 # H, W dimensions
        net = tf.image.resize(net, new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # Or BILINEAR
        return net

    def call(self, fnet_input, training=False): # Add training flag if layers behave differently (e.g. BN, Dropout)
        # Encoder
        enc1 = self._down_block(fnet_input, 32, self.enc_conv1_1, self.enc_lrelu1_1, self.enc_conv1_2, self.enc_lrelu1_2, self.enc_pool1, 'enc1')
        enc2 = self._down_block(enc1, 64, self.enc_conv2_1, self.enc_lrelu2_1, self.enc_conv2_2, self.enc_lrelu2_2, self.enc_pool2, 'enc2')
        enc3 = self._down_block(enc2, 128, self.enc_conv3_1, self.enc_lrelu3_1, self.enc_conv3_2, self.enc_lrelu3_2, self.enc_pool3, 'enc3')

        # Decoder
        dec1 = self._up_block(enc3, 256, self.dec_conv1_1, self.dec_lrelu1_1, self.dec_conv1_2, self.dec_lrelu1_2, 'dec1')
        dec2 = self._up_block(dec1, 128, self.dec_conv2_1, self.dec_lrelu2_1, self.dec_conv2_2, self.dec_lrelu2_2, 'dec2')
        net1 = self._up_block(dec2, 64,  self.dec_conv3_1, self.dec_lrelu3_1, self.dec_conv3_2, self.dec_lrelu3_2, 'dec3') # net1 in original

        # Output stage
        net = self.out_conv1(net1, 3, 32, 1, scope='out_conv1')
        net = self.out_lrelu1(net, 0.2)
        net2 = self.out_conv2(net, 3, 2, 1, scope='out_conv2') # Output 2 channels for flow (x, y)
        net = tf.tanh(net2) * 24.0 # Max velocity
        return net

class Generator(tf.keras.Model):
    def __init__(self, num_resblock=10, vsr_scale=4, gen_output_channels=3, name='generator', **kwargs):
        super(Generator, self).__init__(name=name, **kwargs)
        self.num_resblock = num_resblock
        self.vsr_scale = vsr_scale
        self.gen_output_channels = gen_output_channels

        # Input stage
        self.input_conv = ops.conv2 # To be called with params

        # Residual blocks (store layer ops, not instantiated layers yet if params vary)
        self.res_blocks_conv1 = [ops.conv2 for _ in range(num_resblock)]
        self.res_blocks_conv2 = [ops.conv2 for _ in range(num_resblock)]

        # Decoder / Upsampling
        if self.vsr_scale == 2:
            self.up_conv1 = ops.conv2_tran
        elif self.vsr_scale == 4:
            self.up_conv1 = ops.conv2_tran
            self.up_conv2 = ops.conv2_tran
        
        # Output stage
        self.output_conv_final = ops.conv2 # To be called with params

    def _residual_block(self, inputs, conv1_op, conv2_op, output_channel=64, stride=1, scope_suffix='res_block'):
        net = conv1_op(inputs, 3, output_channel, stride, use_bias=True, scope=f'conv_1_{scope_suffix}')
        net = tf.nn.relu(net)
        net = conv2_op(net, 3, output_channel, stride, use_bias=True, scope=f'conv_2_{scope_suffix}')
        net = net + inputs
        return net

    def call(self, gen_inputs, training=False): # gen_inputs is concat of current_lr and s2d_warped_hr
        # Input stage
        net = self.input_conv(gen_inputs, 3, 64, 1, scope='input_conv')
        stage1_output = tf.nn.relu(net)
        net = stage1_output

        # Residual blocks
        for i in range(self.num_resblock):
            net = self._residual_block(net, self.res_blocks_conv1[i], self.res_blocks_conv2[i], 64, 1, scope_suffix=f'resblock_{i+1}')

        # Decoder / Upsampling (conv_tran2highres)
        if self.vsr_scale == 2:
            net = self.up_conv1(net, kernel=3, output_channel=64, stride=2, scope='up_conv1')
            net = tf.nn.relu(net)
        elif self.vsr_scale == 4:
            net = self.up_conv1(net, 3, 64, 2, scope='up_conv1')
            net = tf.nn.relu(net)
            net = self.up_conv2(net, 3, 64, 2, scope='up_conv2')
            net = tf.nn.relu(net)
        
        # Output stage
        net = self.output_conv_final(net, 3, self.gen_output_channels, 1, scope='output_final_conv')
        
        # Add bicubic of LR input
        # gen_inputs shape: [batch, lr_h, lr_w, lr_c + s2d_hr_c]
        # We need the original LR image part from gen_inputs
        # Assuming LR is the first 3 channels if input is RGB
        low_res_in = gen_inputs[:, :, :, 0:3] # This assumes the LR part is channels 0,1,2
        
        bicubic_hi = ops.bicubic_x(low_res_in, scale=self.vsr_scale)
        net = net + bicubic_hi
        
        # The original generator_f_decoder calls ops.preprocess(net) at the end.
        # This seems unusual if the output is meant to be an image in [0,1] or [-1,1] after deprocess.
        # The TecoGAN paper implies the output is image-like.
        # ops.preprocess shifts to [-1,1]. If bicubic_hi is [0,1] and net from conv is small,
        # then net + bicubic_hi is largely [0,1]. Preprocessing it then makes it [-1,1].
        # The inference script then calls ops.deprocess. This is consistent.
        net = ops.preprocess(net) # Ensure output is in [-1,1] for consistency with inference loop
        return net


# Helper functions to instantiate models (optional, but can simplify main script)
def get_fnet_model():
    return FNet()

def get_generator_model(num_resblock=10, vsr_scale=4, gen_output_channels=3):
    return Generator(num_resblock=num_resblock, vsr_scale=vsr_scale, gen_output_channels=gen_output_channels)