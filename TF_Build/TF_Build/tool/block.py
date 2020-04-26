import tensorflow as tf
from tflearn.layers.conv import global_avg_pool

import tensorflow_tools as tf_tools

def recurrent_block(x, out_channel, name, kernel_size=3, times=2):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for time in range(times):
            if time == 0:
                recurrent_x = tf_tools.conv2d(x, out_channel=out_channel, kernel_size=kernel_size, activation='relu', normal=True, name='recurrent_block')
            
            recurrent_x = tf_tools.conv2d(x + recurrent_x, out_channel=out_channel, kernel_size=kernel_size, activation='relu', normal=True, name='recurrent_block')
        return recurrent_x

def rrcnn_block(x, name, out_channel, kernel_size=3, times=2):
    conv1 = tf_tools.conv2d(x, out_channel=out_channel, kernel_size=1, name=name + '/conv1')
    recurrent_block1 = recurrent_block(conv1, out_channel=out_channel, kernel_size=kernel_size, name=name + '/recurrent_block1')
    recurrent_block2 = recurrent_block(recurrent_block1, kernel_size=kernel_size, out_channel=out_channel, name=name + '/recurrent_block2')
    return conv1 + recurrent_block2

def cnn_block(x, name, out_channel, times=2):
    conv2 = tf_tools.conv2d(x, out_channel=out_channel, activation='relu', normal=True, name=name + '/conv2')
    conv3 = tf_tools.conv2d(conv2, out_channel=out_channel, activation='relu', normal=True, name=name + '/conv3')
    
    return conv3

def dilated_block(x, name, out_channel, rate=2, times=2):
    conv2 = tf_tools.atrous_conv2d(x, rate=rate, out_channel=out_channel, activation='relu', normal=True, name=name + '/conv2')
    conv3 = tf_tools.atrous_conv2d(conv2, rate=rate, out_channel=out_channel, activation='relu', normal=True, name=name + '/conv3')
    
    return conv3

def attention_block(x, last_x, out_isze, name=''):
    with tf.variable_scope("attention_" + name):
        conv_x = tf_tools.conv2d(x, out_channel=out_isze, kernel_size=1, name='conv_x', normal=True)
        conv_last_x = tf_tools.conv2d(last_x, out_channel=out_isze, kernel_size=1, name='conv_last_x', normal=True)
        add_act = tf.nn.relu(conv_x + conv_last_x)
        act = tf_tools.conv2d(add_act, out_channel=1, kernel_size=1, activation='sigmoid', normal=True)
        output = last_x * act
        return output

def Fully_connected(x, units=2, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = tf.nn.relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = tf.nn.sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        scale = input_x * excitation

        return scale

def small_se_res_block(x, out_channel, name='', stride=1, normal=True):
    with tf.variable_scope(name):
        conv1 = tf_tools.conv2d(x, out_channel=out_channel, kernel_sizes=[1, 3], strides=[1, 1, stride, 1], activation='relu', normal=normal, name='conv1')
        conv2 = tf_tools.conv2d(conv1, out_channel=out_channel, kernel_sizes=[3, 1], strides=[1, stride, 1, 1], activation='relu', normal=normal, name='conv2')
        conv3 = tf_tools.conv2d(conv2, out_channel=out_channel, kernel_sizes=[1, 3], activation='relu', normal=normal, name='conv3')
        conv4 = tf_tools.conv2d(conv3, out_channel=out_channel, kernel_sizes=[3, 1], activation='relu', normal=normal, name='conv4')
        se_layer = squeeze_excitation_layer(conv4, out_channel, 4, 'SE')
        conv_x = x
        if stride != 1:
            conv_x = tf_tools.conv2d(x, out_channel=out_channel, kernel_size=1, strides=[1, stride, stride, 1], name='conv_x')
        add = tf.add(conv_x, se_layer)
        output = tf.nn.relu(add)
        return output