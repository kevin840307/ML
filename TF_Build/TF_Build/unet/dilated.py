import sys, os
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

from module import Model
import tensorflow as tf
import tensorflow_tools as tf_tools
from block import cnn_block, dilated_block

class Dilated(Model):
    def __init__(self, scope_name, channel_min, width, height, channel_rate=2):
        super(Dilated, self).__init__(scope_name)
        self.scope_name = scope_name
        self.channel_min = channel_min
        self.channel_rate = channel_rate
        self.input_img = tf.placeholder(tf.float32, shape=[None, width, height, 3], name="input_img")
        self.input_seg = tf.placeholder(tf.float32, shape=[None, width, height, 1], name="input_seg")


    def buind(self, train_fn=tf_tools.adam_fn, lr=0.0001):
        with tf.variable_scope(self.scope_name) as scope:
        
            self.logit_op, self.act_op = self.buind_network()
            self.var_list = tf.trainable_variables(scope=self.scope_name)
            self.index = tf.Variable(0)
            self.train_fn = train_fn(lr)
            
            self.build_optimization()
               
    def buind_network(self):
        with tf.variable_scope("encoder"):
            out_channel = self.channel_min
            cnn1 = cnn_block(self.input_img, out_channel=out_channel, name='cnn1')

            out_channel = out_channel * self.channel_rate
            cnn2 = dilated_block(cnn1, rate=2, out_channel=out_channel, name='cnn2')

            out_channel = out_channel * self.channel_rate
            cnn3 = dilated_block(cnn2, rate=3, out_channel=out_channel, name='cnn3')

            out_channel = out_channel * self.channel_rate
            cnn4 = dilated_block(cnn3, rate=4, out_channel=out_channel, name='cnn4')
            
            out_channel = out_channel * self.channel_rate
            encoder_output = dilated_block(cnn4, rate=5, out_channel=out_channel, name='encoder_output')

        with tf.variable_scope("decoder"):
            out_channel = int(out_channel / self.channel_rate)
            conv1_1 = tf_tools.conv2d(encoder_output, out_channel=out_channel, activation='relu', normal=True, name='conv1_1')
            cnn1 = dilated_block(conv1_1, rate=4, out_channel=out_channel, name='cnn1')

            out_channel = int(out_channel / self.channel_rate)
            conv2_1 = tf_tools.conv2d(cnn1, out_channel=out_channel, activation='relu', normal=True, name='conv2_1')
            cnn2 = dilated_block(conv2_1, rate=3, out_channel=out_channel, name='cnn2')

            out_channel = int(out_channel / self.channel_rate)
            conv3_1 = tf_tools.conv2d(cnn2, out_channel=out_channel, activation='relu', normal=True, name='conv3_1')
            cnn3 = dilated_block(conv3_1, rate=2, out_channel=out_channel, name='cnn3')

            out_channel = int(out_channel / self.channel_rate)
            conv4_1 = tf_tools.conv2d(cnn3, out_channel=out_channel, activation='relu', normal=True, name='conv4_1')
            cnn4 = cnn_block(conv4_1, out_channel=out_channel, name='cnn4')
        
            logit = tf_tools.conv2d(cnn4, out_channel=1, normal=True, name='logit')
            act = tf.nn.sigmoid(logit)

        return logit, act
            
    def build_optimization(self):
        #self.loss_op = total_loss = 1.7 * tf_tools.focal_loss(self.act_op, self.input_seg, alpha=1) + tf_tools.dice_loss(self.act_op, self.input_seg)
        self.loss_op = tf_tools.cross_entropy_loss(self.act_op, self.input_seg)
        self.train_op = self.train_fn.minimize(self.loss_op, global_step=self.index)
              
    def predict(self, x):
        session = tf.get_default_session()
        feed_dict = {self.input_img: x}
        output = session.run(self.act_op, feed_dict=feed_dict)
        return output
    
    
    def train(self, x, y):
        session = tf.get_default_session()
        feed_dict = {self.input_img: x, self.input_seg: y}
        session.run(self.train_op, feed_dict=feed_dict)
        return self.loss(x, y)
    
    def loss(self, x, y):
        session = tf.get_default_session()
        feed_dict = {self.input_img: x, self.input_seg: y}
        loss = session.run(self.loss_op, feed_dict=feed_dict)
        return loss

def minbatch_dataset(data_path, batch_size, start_index):
    imgs = []
    segs = []
    data_dir = os.listdir(data_path + "/image")
    start_index = start_index % len(data_dir)
    max_index = start_index + batch_size
    if max_index >= len(data_dir):
        max_index = len(data_dir)
    
    for index in range(start_index, max_index, 1):
        img = io.imread(os.path.join(data_path + "/image/" + data_dir[index]), as_gray=False)
        seg = io.imread(os.path.join(data_path + "/label/" + data_dir[index]), as_gray=True)

        if img.max() > 1:
            img = img / 255.
        imgs.append(img)
        segs.append(np.expand_dims(seg, axis=-1))

    return np.array(imgs), np.array(segs)

if __name__ == '__main__':
    import skimage.io as io
    import numpy as np
    import os
    import tool
    network = Dilated("Dilated", 16, 128, 128)
    network.buind(train_fn=tf_tools.adam_fn, lr=1e-4)
    network.complete()
    data_path = "./segmentation"

    test_img = io.imread(os.path.join("D:/VisualStudio/VisualStudioProject/TF_Build/TF_Build/segmentation/test/image/TCGA-HE-7130-01Z-00-DX1_60.tif"), as_gray=False)
    test_seg = io.imread(os.path.join("D:/VisualStudio/VisualStudioProject/TF_Build/TF_Build/segmentation/test/label/TCGA-HE-7130-01Z-00-DX1_60.tif"), as_gray=True)
    test_img = np.expand_dims(test_img, axis=0)
    test_seg = np.expand_dims(np.expand_dims(test_seg, axis=0), axis=-1)

    for step in range(2000):
        img, seg = minbatch_dataset(data_path, 4, step)
        network.train(img, seg)
        if step % 10 == 0:
            pred = network.predict(test_img)
            tool.array_img_save(pred[0], "./save/" + str(step) + ".tif", binary=True)
