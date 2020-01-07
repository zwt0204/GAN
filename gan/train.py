# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2020/1/7 10:35
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import tensorflow as tf
from GAN import GAN
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("D:\mygit\GAN\data\MNIST_data\\")


class TrainGan:

    def __init__(self):
        self.class_graph = tf.Graph()
        self.model_dir = 'D:\mygit\GAN\data\gan'
        self.z_dimensions = 100
        self.batch_size = 16
        self.learning_rate =0.0001
        self.is_training = True
        self.model = GAN(self.z_dimensions, self.batch_size, self.learning_rate, self.is_training)
        self.saver = tf.train.Saver()

    def train(self, epochs=30000):
        z_batch = np.random.normal(-1, 1, size=[self.batch_size, self.z_dimensions])
        real_image_batch = mnist.train.next_batch(self.batch_size)
        real_image_batch = np.reshape(real_image_batch[0], [self.batch_size, 28, 28, 1])
        initer = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(initer)
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            if ckpt is not None and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            for epoch in range(epochs):
                # Update the discriminator
                _, dLoss = sess.run([self.model.trainerD, self.model.d_loss],
                                    feed_dict={self.model.z_placeholder: z_batch, self.model.x_placeholder: real_image_batch})
                # Update the generator
                _, gLoss = sess.run([self.model.trainerG, self.model.g_loss], feed_dict={self.model.z_placeholder: z_batch})
                if epoch % 200 == 0:
                    print('判别网络损失%s, 生成网络损失%s' % (dLoss, gLoss))
                self.saver.save(sess, os.path.join(self.model_dir, "gan.dat"))


if __name__ == '__main__':
    train = TrainGan()
    train.train()