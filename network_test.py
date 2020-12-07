import tensorflow as tf
import numpy as np
# from ops import *
from architecture import *
from tensorflow.contrib.framework.python.ops import arg_scope

class network():
    def __init__(self, perturbed_img, perturbed_edge, mask, is_training=False, reuse=False):
        self.is_training = is_training
        self.perturbed_img = perturbed_img  # ground truth
        self.perturbed_edge = perturbed_edge
        self.mask = mask
        self.reuse = reuse

    # structure of the model
    def build_model(self):
        self.generated_edge = self.completion_net_edge(self.perturbed_edge, 1, name='completion_net_edge', is_training=self.is_training, reuse=self.reuse)
        self.reconstruct_edge = (1 - self.mask) * self.perturbed_edge + self.mask * self.generated_edge

        self.generated_img_coarse, self.generated_img_fine = self.completion_net_img(self.perturbed_img, self.reconstruct_edge, name="completion_net_img", is_training=self.is_training, reuse=self.reuse)
        self.reconstruct_img = (1 - self.mask) * self.perturbed_img + self.mask * self.generated_img_fine
        self.reconstruct_img_coarse = (1 - self.mask) * self.perturbed_img + self.mask * self.generated_img_coarse



    # completion network
    def completion_net_edge(self, input, output_channel=1, n_channel=64, name="generator", reuse=False, is_training=True):
        with tf.variable_scope(name, reuse=reuse):
            with arg_scope([conv, deconv], normalizer_fn=batch_norm):
                with arg_scope([batch_norm], is_training=is_training):
                    ones_input = tf.ones_like(input)[:, :, :, 0:1]
                    input = tf.concat([input, ones_input, ones_input * self.mask], axis=3)
                    conv1 = conv(input, n_channel * 1, 5, 1, scope="conv1")
                    conv2 = conv(conv1, n_channel * 2, 3, 2, scope="conv2")
                    conv3 = conv(conv2, n_channel * 2, 3, 1, scope="conv3")
                    conv4 = conv(conv3, n_channel * 4, 3, 2, scope="conv4")
                    conv5 = conv(conv4, n_channel * 4, 3, 1, scope="conv5")
                    conv6 = conv(conv5, n_channel * 4, 3, 1, scope="conv6")

                    conv7 = conv(conv6, n_channel * 4, 3, 1, rate=2, scope="dilate_conv7")
                    conv8 = conv(conv7, n_channel * 4, 3, 1, rate=4, scope="dilate_conv8")
                    conv9 = conv(conv8, n_channel * 4, 3, 1, rate=8, scope="dilate_conv9")
                    conv10 = conv(conv9, n_channel * 4, 3, 1, rate=16, scope="dilate_conv10")

                    conv11 = conv(conv10, n_channel * 4, 3, 1, scope="conv11")
                    conv12 = conv(conv11, n_channel * 4, 3, 1, scope="conv12")
                    conv13 = deconv(conv12, n_channel * 2, scope="deconv13")
                    conv14 = conv(conv13, n_channel * 2, 3, 1, scope="conv14")
                    conv15 = deconv(conv14, n_channel * 1, scope="deconv15")
                    conv16 = conv(conv15, n_channel // 2, 3, 1, scope="conv16")
                    conv16 = tf.concat([conv16, input], -1)
                    conv17 = conv(conv16, output_channel, 3, 1, scope="conv17", activation_fn=tf.nn.tanh)

                    return conv17

    def completion_net_img(self, input_img, input_edge, n_channel=32, name="generator", reuse=False, is_training=True):
        with tf.variable_scope(name, reuse=reuse):
            # stage 1
            ones_input = tf.ones_like(input_img)[:, :, :, 0:1]

            img_stream = tf.concat([input_img, ones_input, ones_input * self.mask], axis=3)
            img_stream = conv(img_stream, n_channel, 5, 1, scope="conv1_1_img")
            img_stream = conv(img_stream, n_channel, 3, 1, scope="conv1_2_img")
            img_stream = conv(img_stream, n_channel, 3, 1, scope="conv1_3_img")
            img_stream = conv(img_stream, n_channel, 3, 1, scope="conv1_4_img")

            edge_stream = tf.concat([input_edge, ones_input, ones_input * self.mask], axis=3)
            edge_stream = conv(edge_stream, n_channel, 5, 1, scope="conv1_1_edge")
            edge_stream = conv(edge_stream, n_channel, 3, 1, scope="conv1_2_edge")
            edge_stream = conv(edge_stream, n_channel, 3, 1, scope="conv1_3_edge")
            edge_stream = conv(edge_stream, n_channel, 3, 1, scope="conv1_4_edge")

            input_concate1 = tf.concat([img_stream, edge_stream], -1)

            conv1_1 = conv(input_concate1, n_channel * 2, 3, 1, scope="conv1_1")
            conv1_2 = conv(conv1_1, n_channel * 2, 3, 2, scope="conv1_2")
            conv1_3 = conv(conv1_2, n_channel * 2, 3, 1, scope="conv1_3")
            conv1_4 = conv(conv1_3, n_channel * 2, 3, 2, scope="conv1_4")
            conv1_5 = conv(conv1_4, n_channel * 4, 3, 1, scope="conv1_5")
            conv1_6 = conv(conv1_5, n_channel * 4, 3, 1, scope="conv1_6")
            mask_s = resize_mask_like(self.mask, conv1_6)
            conv1_7 = conv(conv1_6, n_channel * 4, 3, 1, rate=2, scope="dilate_conv1_7")
            conv1_8 = conv(conv1_7, n_channel * 4, 3, 1, rate=4, scope="dilate_conv1_8")
            conv1_9 = conv(conv1_8, n_channel * 4, 3, 1, rate=8, scope="dilate_conv1_9")
            conv1_10 = conv(conv1_9, n_channel * 4, 3, 1, rate=16, scope="dilate_conv1_10")
            conv1_11 = conv(conv1_10, n_channel * 4, 3, 1, scope="conv1_11")
            conv1_12 = conv(conv1_11, n_channel * 4, 3, 1, scope="conv1_12")
            conv1_13 = deconv(conv1_12, n_channel * 2, scope="deconv1_13")
            conv1_14 = conv(conv1_13, n_channel * 2, 3, 1, scope="conv1_14")
            conv1_15 = deconv(conv1_14, n_channel * 1, scope="deconv1_15")
            conv1_16 = conv(conv1_15, n_channel // 2, 3, 1, scope="conv1_16")
            conv1_16 = tf.concat([conv1_16, input_img, ones_input, ones_input * self.mask], -1)
            conv1_17 = conv(conv1_16, 3, 3, 1, scope="conv1_17", activation_fn=None)
            conv1_17 = tf.clip_by_value(conv1_17, -1., 1.)

            # stage 2
            img_stream = tf.concat([conv1_17, ones_input, ones_input * self.mask], axis=3)
            img_stream = conv(img_stream, n_channel, 5, 1, scope="conv2_1_img")
            img_stream = conv(img_stream, n_channel, 3, 1, scope="conv2_2_img")
            img_stream = conv(img_stream, n_channel, 3, 1, scope="conv2_3_img")
            img_stream = conv(img_stream, n_channel, 3, 1, scope="conv2_4_img")

            edge_stream = tf.concat([input_edge, ones_input, ones_input * self.mask], axis=3)
            edge_stream = conv(edge_stream, n_channel, 5, 1, scope="conv2_1_edge")
            edge_stream = conv(edge_stream, n_channel, 3, 1, scope="conv2_2_edge")
            edge_stream = conv(edge_stream, n_channel, 3, 1, scope="conv2_3_edge")
            edge_stream = conv(edge_stream, n_channel, 3, 1, scope="conv2_4_edge")

            input_concate2 = tf.concat([img_stream, edge_stream], -1)

            # conv branch
            conv2_1 = conv(input_concate2, n_channel * 2, 3, 1, scope="conv2_1")
            conv2_2 = conv(conv2_1, n_channel * 2, 3, 2, scope="conv2_2")
            conv2_3 = conv(conv2_2, n_channel * 2, 3, 1, scope="conv2_3")
            conv2_4 = conv(conv2_3, n_channel * 2, 3, 2, scope="conv2_4")
            conv2_5 = conv(conv2_4, n_channel * 4, 3, 1, scope="conv2_5")
            conv2_6 = conv(conv2_5, n_channel * 4, 3, 1, scope="conv2_6")
            conv2_7 = conv(conv2_6, n_channel * 4, 3, 1, rate=2, scope="dilate_conv2_7")
            conv2_8 = conv(conv2_7, n_channel * 4, 3, 1, rate=4, scope="dilate_conv2_8")
            conv2_9 = conv(conv2_8, n_channel * 4, 3, 1, rate=8, scope="dilate_conv2_9")
            conv2_10 = conv(conv2_9, n_channel * 4, 3, 1, rate=16, scope="dilate_conv2_10")
            # ca branch
            conv2_1_ca = conv(input_concate2, n_channel * 2, 5, 1, scope="conv2_1_ca")
            conv2_2_ca = conv(conv2_1_ca, n_channel * 2, 3, 2, scope="conv2_2_ca")
            conv2_3_ca = conv(conv2_2_ca, n_channel * 2, 3, 1, scope="conv2_3_ca")
            conv2_4_ca = conv(conv2_3_ca, n_channel * 2, 3, 2, scope="conv2_4_ca")
            conv2_5_ca = conv(conv2_4_ca, n_channel * 4, 3, 1, scope="conv2_5_ca")
            conv2_6_ca = conv(conv2_5_ca, n_channel * 4, 3, 1, scope="conv2_6_ca", activation_fn=tf.nn.relu)
            ca = contextual_attention(conv2_6_ca, conv2_6_ca, mask_s, 3, 1, rate=2)
            conv2_7_ca = conv(ca, n_channel * 4, 3, 1, scope="conv2_7_ca")
            conv2_8_ca = conv(conv2_7_ca, n_channel * 4, 3, 1, scope="conv2_8_ca")
            # merge 2 branches
            merge = tf.concat([conv2_10, conv2_8_ca], axis=3)
            conv2_11 = conv(merge, n_channel * 4, 3, 1, scope="conv2_11")
            conv2_12 = conv(conv2_11, n_channel * 4, 3, 1, scope="conv2_12")
            conv2_13 = deconv(conv2_12, n_channel * 2, scope="deconv2_13")
            conv2_14 = conv(conv2_13, n_channel * 2, 3, 1, scope="conv2_14")
            conv2_15 = deconv(conv2_14, n_channel * 1, scope="deconv2_15")
            conv2_16 = conv(conv2_15, n_channel // 2, 3, 1, scope="conv2_16")
            conv2_16 = tf.concat([conv2_16, input_img, ones_input, ones_input * self.mask], -1)
            conv2_17 = conv(conv2_16, 3, 3, 1, scope="conv2_17", activation_fn=None)
            conv2_17 = tf.clip_by_value(conv2_17, -1., 1.)

            return conv1_17, conv2_17
