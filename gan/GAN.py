# -*- encoding: utf-8 -*-
"""
@File    : GAN.py
@Time    : 2020/1/7 10:24
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import tensorflow as tf
tf.reset_default_graph()


class GAN:

    def __init__(self, z_dimensions, batch_size, learning_rate, is_training):
        self.z_dimensions = z_dimensions
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_training = is_training
        # 真实图片
        # Placeholder for input images to the discriminator
        self.x_placeholder = tf.placeholder("float", shape=[None, 28, 28, 1])
        # 噪声
        # Placeholder for input noise vectors to the generator
        self.z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])
        self.creat_modle()
        if self.is_training is True:
            self.create_loss()

    def creat_modle(self):
        self.Dx = self.discriminator(self.x_placeholder)
        self.Gz = self.generator(self.z_placeholder, self.batch_size, self.z_dimensions)
        self.Dg = self.discriminator(self.Gz, reuse=True)

    def create_loss(self):
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dg, labels=tf.ones_like(self.Dg)))
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dx, labels=tf.ones_like(self.Dx)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dg, labels=tf.zeros_like(self.Dg)))
        self.d_loss = d_loss_real + d_loss_fake

        tvars = tf.trainable_variables()
        d_vars = [var for var in tvars if 'd_' in var.name]
        g_vars = [var for var in tvars if 'g_' in var.name]
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            self.trainerD = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss, var_list=d_vars)
            self.trainerG = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss, var_list=g_vars)

    def conv2d(self, x, W):
        return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

    def avg_pool_2x2(self, x):
        return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def discriminator(self, x_image, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if (reuse):
                tf.get_variable_scope().reuse_variables()
            # First Conv and Pool Layers
            W_conv1 = tf.get_variable('d_wconv1', [5, 5, 1, 8],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv1 = tf.get_variable('d_bconv1', [8], initializer=tf.constant_initializer(0))
            h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = self.avg_pool_2x2(h_conv1)

            # Second Conv and Pool Layers
            W_conv2 = tf.get_variable('d_wconv2', [5, 5, 8, 16],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv2 = tf.get_variable('d_bconv2', [16], initializer=tf.constant_initializer(0))
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = self.avg_pool_2x2(h_conv2)

            # First Fully Connected Layer
            W_fc1 = tf.get_variable('d_wfc1', [7 * 7 * 16, 32],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_fc1 = tf.get_variable('d_bfc1', [32], initializer=tf.constant_initializer(0))
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 16])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            # Second Fully Connected Layer
            W_fc2 = tf.get_variable('d_wfc2', [32, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_fc2 = tf.get_variable('d_bfc2', [1], initializer=tf.constant_initializer(0))

            # Final Layer
            y_conv = (tf.matmul(h_fc1, W_fc2) + b_fc2)
        return y_conv

    def generator(self, z, batch_size, z_dim):
        with tf.variable_scope('generator') as scope:
            # Number of filters of first layer of generator
            g_dim = 64
            # Color dimension of output (MNIST is grayscale, so c_dim = 1 for us)
            c_dim = 1
            # Output size of the image
            s = 28
            # We want to slowly upscale the image, so these values will help
            # make that change gradual.
            s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(
                s / 16)
            h0 = tf.reshape(z, [batch_size, s16 + 1, s16 + 1, 25])
            h0 = tf.nn.relu(h0)
            # Dimensions of h0 = batch_size x 2 x 2 x 25

            # First DeConv Layer
            output1_shape = [batch_size, s8, s8, g_dim * 4]
            W_conv1 = tf.get_variable('g_wconv1', [5, 5, output1_shape[-1], int(h0.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(.1))
            H_conv1 = tf.nn.conv2d_transpose(h0, W_conv1, output_shape=output1_shape, strides=[1, 2, 2, 1],
                                             padding='SAME')
            H_conv1 = tf.contrib.layers.batch_norm(inputs=H_conv1, center=True, scale=True, is_training=True,
                                                   scope="g_bn1")
            H_conv1 = tf.nn.relu(H_conv1)
            # Dimensions of H_conv1 = batch_size x 3 x 3 x 256

            # Second DeConv Layer
            output2_shape = [batch_size, s4 - 1, s4 - 1, g_dim * 2]
            W_conv2 = tf.get_variable('g_wconv2', [5, 5, output2_shape[-1], int(H_conv1.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(.1))
            H_conv2 = tf.nn.conv2d_transpose(H_conv1, W_conv2, output_shape=output2_shape, strides=[1, 2, 2, 1],
                                             padding='SAME')
            H_conv2 = tf.contrib.layers.batch_norm(inputs=H_conv2, center=True, scale=True, is_training=True,
                                                   scope="g_bn2")
            H_conv2 = tf.nn.relu(H_conv2)
            # Dimensions of H_conv2 = batch_size x 6 x 6 x 128

            # Third DeConv Layer
            output3_shape = [batch_size, s2 - 2, s2 - 2, g_dim * 1]
            W_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
            H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape, strides=[1, 2, 2, 1],
                                             padding='SAME')
            H_conv3 = tf.contrib.layers.batch_norm(inputs=H_conv3, center=True, scale=True, is_training=True,
                                                   scope="g_bn3")
            H_conv3 = tf.nn.relu(H_conv3)
            # Dimensions of H_conv3 = batch_size x 12 x 12 x 64

            # Fourth DeConv Layer
            output4_shape = [batch_size, s, s, c_dim]
            W_conv4 = tf.get_variable('g_wconv4', [5, 5, output4_shape[-1], int(H_conv3.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv4 = tf.get_variable('g_bconv4', [output4_shape[-1]], initializer=tf.constant_initializer(.1))
            H_conv4 = tf.nn.conv2d_transpose(H_conv3, W_conv4, output_shape=output4_shape, strides=[1, 2, 2, 1],
                                             padding='VALID')
            H_conv4 = tf.nn.tanh(H_conv4)
            # Dimensions of H_conv4 = batch_size x 28 x 28 x 1

        return H_conv4