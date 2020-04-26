import sys, os
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

from module import Model
from discriminator import Discriminator
from generator import Generator
import tensorflow as tf
import tensorflow_tools as tf_tools

class WGAN(Model):
    def __init__(self, scope_name, channel_min, img_size, generator_size=100, channel_rate=2):
        super(WGAN, self).__init__(scope_name)
        self.scope_name = scope_name
        self.channel_min = channel_min
        self.channel_rate = channel_rate
        self.img_size = img_size
        self.input_img = tf.placeholder(tf.float32, shape=[None] + img_size, name="input_img")
        self.input_z = tf.placeholder(tf.float32, shape=[None, generator_size], name="input_z")

    def buind(self, train_fn=tf_tools.adam_fn, real_lr=1e-5, fake_lr=1e-5):
        with tf.variable_scope(self.scope_name) as scope:
            self.fake_img, self.real_output, self.fake_output = self.buind_network()
            self.var_list = tf.trainable_variables(scope=self.scope_name)
            self.real_train_fn = train_fn(real_lr, 0, 0.9)
            self.fake_train_fn = train_fn(fake_lr, 0, 0.9)
            self.build_optimization()
            
    
    def buind_network(self, fake_normal=True):
        self.real_network = Discriminator(self.channel_min, 1, name="discriminator")
        self.fake_network = Generator(self.channel_min, self.img_size, name="generator")

        fake_img = self.fake_network.build(self.input_z, times=3)
        if fake_normal:
            fake_img = tf.nn.sigmoid(fake_img)

        real_output =  self.real_network.build(self.input_img, times=3, normal=False)
        fake_output = self.real_network.build(fake_img, times=3, reuse=tf.AUTO_REUSE, normal=False)

        return fake_img, real_output, fake_output
            
    def build_optimization(self):
        self.real_loss_op =  tf.reduce_mean(self.fake_output) - tf.reduce_mean(self.real_output)
        self.fake_loss_op = -tf.reduce_mean(self.fake_output)

        self.real_index = tf.Variable(0)
        self.real_train_op = self.real_train_fn.minimize(self.real_loss_op, var_list=self.real_network.var_list, global_step=self.real_index)
        self.clip_op = [param.assign(tf.clip_by_value(param, -0.01, 0.01)) for param in self.real_network.var_list]

        self.fake_index = tf.Variable(0)
        self.fake_train_op = self.fake_train_fn.minimize(self.fake_loss_op, var_list=self.fake_network.var_list, global_step=self.fake_index)
        
        
    def predict(self, z):
        session = tf.get_default_session()
        feed_dict = {self.input_z: z}
        output = session.run(self.fake_img, feed_dict=feed_dict)
        return output
    
    
    def train(self, img, z, mode='D'):
        session = tf.get_default_session()
        feed_dict = {self.input_img: img, self.input_z: z}
        if mode == 'D':
            session.run(self.clip_op, feed_dict=feed_dict)
            session.run(self.real_train_op, feed_dict=feed_dict)
        else:
            session.run(self.fake_train_op, feed_dict=feed_dict)
        return self.loss(img, z)
    
    def loss(self, img, z):
        session = tf.get_default_session()
        feed_dict = {self.input_img: img, self.input_z: z}
        loss = session.run([self.real_loss_op, self.fake_loss_op], feed_dict=feed_dict)
        return loss

if __name__ == '__main__':
    import skimage.io as io
    import numpy as np
    import os
    import tool
    from tensorflow.examples.tutorials.mnist import input_data
    import tensorflow_tools as tf_tools

    batch_size = 64
    generator_size = 100
    img_size = [28, 28, 1]
    network = WGAN("WGAN", 64, generator_size=generator_size, img_size=img_size)
    network.buind(train_fn=tf_tools.adam_fn, real_lr=5e-5, fake_lr=5e-5)
    network.complete()

    mnist = input_data.read_data_sets("MNIST/", one_hot=True)

    for step in range(500):
        img, _ = mnist.train.next_batch(batch_size)
        img = np.reshape(img, [-1] + img_size)
        sample_z = np.random.uniform(-1., 1., (batch_size, generator_size))

        network.train(img, sample_z, mode='D')

    for step in range(1000000):
        img, _ = mnist.train.next_batch(batch_size)
        img = np.reshape(img, [-1] + img_size)
        sample_z = np.random.uniform(-1., 1., (batch_size, generator_size))

        network.train(img, sample_z, mode='D')
        loss = network.train(img, sample_z, mode='G')
        if step % 20 == 0:
            print(loss)
            pred = network.predict(sample_z)
            tool.array_img_save(pred[0], "./save/" + str(step) + ".tif", binary=True)
