from module import Model
import tensorflow as tf
import tensorflow_tools as tf_tools
from block import cnn_block, rrcnn_block, attention_block

class AttentionR2UNet(Model):
    def __init__(self, scope_name, channel_min, width, height, channel_rate=2):
        super(AttentionR2UNet, self).__init__(scope_name)
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
            rrcnn1 = rrcnn_block(self.input_img, out_channel=out_channel, name='rrcnn1')
            pool1 = tf_tools.conv2d(rrcnn1, strides=[1, 2, 2, 1], out_channel=out_channel, activation='relu', normal=True, name='pool1')

            out_channel = out_channel * self.channel_rate
            rrcnn2 = rrcnn_block(pool1, out_channel=out_channel, name='rrcnn2')
            pool2 = tf_tools.conv2d(rrcnn2, strides=[1, 2, 2, 1], out_channel=out_channel, activation='relu', normal=True, name='pool2')

            out_channel = out_channel * self.channel_rate
            rrcnn3 = rrcnn_block(pool2, out_channel=out_channel, name='rrcnn3')
            pool3 = tf_tools.conv2d(rrcnn3, strides=[1, 2, 2, 1], out_channel=out_channel, activation='relu', normal=True, name='pool3')

            out_channel = out_channel * self.channel_rate
            rrcnn4 = rrcnn_block(pool3, out_channel=out_channel, name='rrcnn4')
            pool4 = tf_tools.conv2d(rrcnn4, strides=[1, 2, 2, 1], out_channel=out_channel, activation='relu', normal=True, name='pool4')
            
            out_channel = out_channel * self.channel_rate
            encoder_output = cnn_block(pool4, out_channel=out_channel, name='encoder_output')
            concat_list = [rrcnn4, rrcnn3, rrcnn2, rrcnn1]

        with tf.variable_scope("decoder"):
            out_channel = int(out_channel / self.channel_rate)
            shape = concat_list[0].get_shape().as_list()
            deconv1 = tf_tools.upsampling(encoder_output, [shape[1], shape[2]])
            conv1_1 = tf_tools.conv2d(deconv1, out_channel=out_channel, activation='relu', normal=True, name='conv1_1')
            attention1 = attention_block(conv1_1, concat_list[0], out_isze=out_channel, name='attention1')
            concat1 = tf.concat([concat_list[0], conv1_1], axis=-1)
            rrcnn1 = rrcnn_block(concat1, out_channel=out_channel, name='rrcnn1')

            out_channel = int(out_channel / self.channel_rate)
            shape = concat_list[1].get_shape().as_list()
            deconv2 = tf_tools.upsampling(rrcnn1, [shape[1], shape[2]])
            conv2_1 = tf_tools.conv2d(deconv2, out_channel=out_channel, activation='relu', normal=True, name='conv2_1')
            attention2 = attention_block(conv2_1, concat_list[1], out_isze=out_channel, name='attention2')
            concat2 = tf.concat([attention2, conv2_1], axis=-1)
            rrcnn2 = rrcnn_block(concat2, out_channel=out_channel, name='rrcnn2')

            out_channel = int(out_channel / self.channel_rate)
            shape = concat_list[2].get_shape().as_list()
            deconv3 = tf_tools.upsampling(rrcnn2, [shape[1], shape[2]])
            conv3_1 = tf_tools.conv2d(deconv3, out_channel=out_channel, activation='relu', normal=True, name='conv3_1')
            attention3 = attention_block(conv3_1, concat_list[2], out_isze=out_channel, name='attention3')
            concat3 = tf.concat([attention3, conv3_1], axis=-1)
            rrcnn3 = rrcnn_block(concat3, out_channel=out_channel, name='rrcnn3')

            out_channel = int(out_channel / self.channel_rate)
            shape = concat_list[3].get_shape().as_list()
            deconv4 = tf_tools.upsampling(rrcnn3, [shape[1], shape[2]])
            conv4_1 = tf_tools.conv2d(deconv4, out_channel=out_channel, activation='relu', normal=True, name='conv4_1')
            attention4 = attention_block(conv4_1, concat_list[3], out_isze=out_channel, name='attention4')
            concat4 = tf.concat([attention4, conv4_1], axis=-1)
            rrcnn4 = rrcnn_block(concat4, out_channel=out_channel, name='rrcnn4')
        
            logit = tf_tools.conv2d(rrcnn4, out_channel=1, normal=True, name='logit')
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

if __name__ == '__main__':
    import skimage.io as io
    import numpy as np
    import os
    import tool
    network = AttentionR2UNet("AttentionR2UNet", 8, 128, 128)
    network.buind(train_fn=tf_tools.adam_fn, lr=1e-4)
    network.complete()

    img = io.imread(os.path.join("./segmentation/image/08_2_3.tif"), as_gray=False)
    seg = io.imread(os.path.join("./segmentation/label/08_2_3.tif"), as_gray=True)

    img = np.expand_dims(img, axis=0)
    seg = np.expand_dims(np.expand_dims(seg, axis=0), axis=-1)

    #img = np.tile(img, [4, 1, 1, 1])
    #seg = np.tile(seg, [4, 1, 1, 1])

    for step in range(1000):
        network.train(img, seg)
        if step % 10 == 0:
            pred = network.predict(img)
            tool.array_img_save(pred[0], "./save/" + str(step) + ".tif", binary=True)
