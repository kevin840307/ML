from module import Model
import tensorflow as tf
import tensorflow_tools as tf_tools
from block import cnn_block
import numpy as np
import math


class SPPNet(Model):
    def __init__(self, scope_name, class_num, size_list, batch_size, channel_min=32, channel_rate=2):
        super(SPPNet, self).__init__(scope_name)
        assert type(size_list) is np.ndarray, 'SPPNet is an error from the size_list of type'
        self.scope_name = scope_name
        self.channel_min = channel_min
        self.channel_rate = channel_rate
        self.class_num = class_num
        self.batch_size = batch_size
        self.size_list = size_list
        self.input_img_list = [tf.placeholder(tf.float32, shape=[None, size_list[index, 0], size_list[index, 1], 1], name="input_img_{}".format(index)) for index in range(len(size_list))]
        self.input_label = tf.placeholder(tf.float32, shape=[None, 1], name="input_label")

    def buind(self, train_fn=tf_tools.adam_fn, lr=0.0001, seed=None):
        if not seed is None:
            np.random.seed(seed)
            tf.set_random_seed(seed)

        with tf.variable_scope(self.scope_name) as scope:
            
            self.logit_op_list, self.act_op_list = self.buind_network()
            self.var_list = tf.trainable_variables(scope=self.scope_name)
            self.index = tf.Variable(0)
            self.train_fn = train_fn(lr)
            
            self.build_optimization()     
    
    def buind_network(self):
        logit_list = []
        act_list = []
        index = 0
        for input_img in self.input_img_list:
            with tf.variable_scope("predict", reuse=tf.AUTO_REUSE):
                out_channel = self.channel_min
                cnn1 = cnn_block(input_img, out_channel=out_channel, name='cnn1')
                pool1 = tf_tools.max_pool(cnn1, name='pool1')

                out_channel = out_channel * self.channel_rate
                cnn2 = cnn_block(input_img, out_channel=out_channel, name='cnn2')
                pool2 = tf_tools.max_pool(cnn2, name='pool2')

                out_channel = out_channel * self.channel_rate
                cnn3 = cnn_block(input_img, out_channel=out_channel, name='cnn3')
                pool3 = tf_tools.max_pool(cnn3, name='pool3')

                out_channel = out_channel * self.channel_rate
                cnn4 = cnn_block(pool3, out_channel=out_channel, name='cnn4')
                shape = cnn4.get_shape().as_list()
                spp = tf_tools.spatial_pyramid_pool(cnn4, [shape[1], shape[2]], [1, 2, 4], 'spp_{}'.format(index))

                flatten = tf.layers.flatten(spp)
                layer1 = tf_tools.layer(flatten, out_size=4096, activation='relu', normal=True, name='flatten')
                logit = tf_tools.layer(flatten, out_size=self.class_num, name='logit')
                act = tf.argmax(logit, -1)
                act = tf.expand_dims(act, -1)
                act = tf.reshape(act, shape=[-1, 1])
                act = tf.cast(act, tf.float32)

                logit_list.append(logit)
                act_list.append(act)
                index += 1
        return logit_list, act_list
            
    def build_optimization(self):
        loss_op_list = []
        train_op_list = []
        for logit_op in self.logit_op_list:
            label = tf.cast(self.input_label, tf.int32)
            cross = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(label, squeeze_dims=[1]), logits=logit_op)
            loss_op = tf.reduce_mean(cross)
            train_op = self.train_fn.minimize(loss_op, global_step=self.index)
            loss_op_list.append(loss_op)
            train_op_list.append(train_op)
        self.loss_op_list = loss_op_list
        self.train_op_list = train_op_list
        
    def predict(self, x):
        session = tf.get_default_session()
        index = self.size_list.tolist().index([np.shape(x)[1], np.shape(x)[2]])
        feed_dict = {self.input_img_list[index]: x}
        output = session.run(self.act_op_list[index], feed_dict=feed_dict)
        return output
    
    
    def train(self, x, y):
        session = tf.get_default_session()
        index = self.size_list.tolist().index([np.shape(x)[1], np.shape(x)[2]])
        feed_dict = {self.input_img_list[index]: x, self.input_label: y}
        session.run(self.train_op_list[index], feed_dict=feed_dict)
        return self.loss(x, y)
    
    def loss(self, x, y):
        session = tf.get_default_session()
        index = self.size_list.tolist().index([np.shape(x)[1], np.shape(x)[2]])
        feed_dict = {self.input_img_list[index]: x, self.input_label: y}
        loss = session.run(self.loss_op_list[index], feed_dict=feed_dict)
        return loss


import metric
import os
def validation_class(network, dataset, type='validation'):
    accuracy = 0.

    time_max = dataset.get_size(type='validation') // batch_size
    for index in range(0, time_max, 1):
        data_x, data_y = dataset.get_minbatch(batch_size, index, type=type)
        pred_label = network.predict(data_x)
        pred_label = pred_label.astype(np.int)
        pred_label = np.equal(pred_label, data_y).astype(np.int)
        data_y = np.ones_like(pred_label)

        accuracy += metric.accuracy(pred_label, data_y)

    accuracy = accuracy / time_max

    return accuracy


if __name__ == '__main__':
    #from flower_dataset import FlowerDataset
    #batch_size = 50
    #step = 20
    #flower50x50 = FlowerDataset(image_height=50, image_width=50)
    #flower40x40 = FlowerDataset(image_height=40, image_width=40)
    #dpp_net = SPPNet("SPPNet", 102, np.array([[50, 50], [40, 40]]), batch_size)
    #dpp_net.buind()
    #dpp_net.complete()

    #loss = 0.
    #for time in range(1, 10000, 1):
    #    if time % 2 == 0:
    #        imgs, lables = flower50x50.get_minbatch(batch_size, time - 1)
    #    else:
    #        imgs, lables = flower40x40.get_minbatch(batch_size, time - 1)
    #    loss += dpp_net.train(imgs, lables)

    #    if time % step == 0:
    #        accuracy = validation_class(dpp_net, flower)
    #        print('time: %d, loss: %.5f, accuracy: %.3f' % (time, loss / step, accuracy))
    #        loss = 0

    from mnist_dataset import MnistDataset
    batch_size = 100
    step = 100
    mnist28x28 = MnistDataset()
    #mnist14x14 = MnistDataset(image_height=14, image_width=14)
    dpp_net = SPPNet("SPPNet", 10, np.array([[28, 28], [14, 14]]), batch_size, channel_min=1)
    dpp_net.buind(seed=1234)
    dpp_net.complete()

    loss = 0.
    for time in range(1, 10000, 1):
        imgs, lables = mnist28x28.get_minbatch(batch_size, time - 1)
        loss += dpp_net.train(imgs, lables)

        #imgs, lables = mnist14x14.get_minbatch(batch_size, time - 1)
        #loss += dpp_net.train(imgs, lables)
        if time % step == 0:
            accuracy = validation_class(dpp_net, mnist28x28)
            print('time: %d, loss: %.5f, accuracy: %.3f' % (time, loss / step, accuracy))
            loss = 0