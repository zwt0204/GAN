# -*- encoding: utf-8 -*-
"""
@File    : predict.py
@Time    : 2020/1/7 10:56
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import tensorflow as tf
from GAN import GAN
import numpy as np
import logging
import matplotlib.pyplot as plt


class Predict:

    def __init__(self):
        self.graph = tf.Graph()
        self.model_dir = 'D:\mygit\GAN\data\gan'
        self.z_dimensions = 100
        self.batch_size = 1
        self.learning_rate = 0.0001
        self.is_training = False
        with self.graph.as_default():
            self.model = GAN(self.z_dimensions, self.batch_size, self.learning_rate, self.is_training)
            self.saver = tf.train.Saver()

        config = tf.ConfigProto(log_device_placement=False)
        self.session = tf.Session(graph=self.graph, config=config)
        self.load()

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt is not None and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            logging.info("load classification success...")
        else:
            raise Exception("load classification failure...")

    def predict(self):
        # 生成图片
        z_batch = np.random.normal(-1, 1, size=[1, self.model.z_dimensions])
        temp = (self.session.run(self.model.Gz, feed_dict={self.model.z_placeholder: z_batch}))
        my_i = temp.squeeze()
        plt.imshow(my_i, cmap='gray_r')
        plt.show()


if __name__ == '__main__':
    predict = Predict()
    predict.predict()