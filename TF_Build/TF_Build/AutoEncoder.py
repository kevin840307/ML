import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow_tools as tf_tools
from PIL import Image
import skimage.io as io
import skimage.transform as trans
import math
from skimage import color

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


D_learning_rate = 0.00001
G_learning_rate = 0.00001
batch_size = 10
test_size = 5
train_times = 1000000
train_step = 20


image_width = 64
image_height = 64
image_size = image_width * image_height
channel = 3

# [kernel height, filter weight, input channel, out channel]
discriminator_conv1_size = [5, 5, channel * 2, 128]
discriminator_conv2_size = [5, 5, 128, 64]
discriminator_conv3_size = [5, 5, 64, 32]
discriminator_conv4_size = [5, 5, 32, 16]
discriminator_conv5_size = [1, 1, 16, 1]


generator_conv1_size = [5, 5, channel + 1, 128]
generator_conv2_size = [5, 5, 128, 64]
generator_conv3_size = [5, 5, 64, 32]
generator_conv4_size = [5, 5, 32, 16]
generator_conv5_size = [5, 5, 16, 3]

#generator_fiter4_size = 1


def discriminator(x, y, share=False):
    with tf.variable_scope("d_init", reuse=share):
        input_data = tf.concat([x, y], axis=3)

    with tf.variable_scope("discriminator", reuse=share):
        with tf.variable_scope("conv1", reuse=share):
            output = tf_tools.conv2d(input_data, discriminator_conv1_size, strides=[1, 2, 2, 1])
            output = tf_tools.leaky_relu(output)
        with tf.variable_scope("conv2", reuse=share):
            output = tf_tools.conv2d(output, discriminator_conv2_size, strides=[1, 2, 2, 1])
            output = tf_tools.leaky_relu(output)
        with tf.variable_scope("conv3", reuse=share):
            output = tf_tools.conv2d(output, discriminator_conv3_size, strides=[1, 2, 2, 1])
            output = tf_tools.leaky_relu(output)
        with tf.variable_scope("conv4", reuse=share):
            output = tf_tools.conv2d(output, discriminator_conv4_size, strides=[1, 2, 2, 1])
            output = tf_tools.leaky_relu(output)
        with tf.variable_scope("conv5", reuse=share):
            output = tf_tools.conv2d(output, discriminator_conv5_size, strides=[1, 2, 2, 1])
            output = tf_tools.sigmoid(output)
            output = tf.reshape(output, shape=[-1, 1])

    return output

def generator(x, y):

    with tf.variable_scope("g_init"):
        #x = tf.tile(x, [1, 1, 1, channel])
        input_data = tf.concat([x, y], axis=3)

    with tf.variable_scope("generator"):
        with tf.variable_scope("conv1"):
            output = tf_tools.conv2d(input_data, generator_conv1_size)
            output = tf_tools.leaky_relu(output)
        with tf.variable_scope("conv2"):
            output = tf_tools.conv2d(output, generator_conv2_size)
            output = tf_tools.leaky_relu(output)
        with tf.variable_scope("conv3"):
            output = tf_tools.conv2d(output, generator_conv3_size)
            output = tf_tools.leaky_relu(output)
        with tf.variable_scope("conv4"):
            output = tf_tools.conv2d(output, generator_conv4_size)
            output = tf_tools.leaky_relu(output)
        with tf.variable_scope("conv5"):
            output = tf_tools.conv2d(output, generator_conv5_size)
            output = tf_tools.tanh(output)
    return output

def discriminator_loss(D_x, D_G):
    loss =  -tf.reduce_mean(tf.log(D_x + 1e-12) + tf.log(1. - D_G + 1e-12))
    loss_his = tf.summary.scalar("discriminator_loss", loss)

    return loss

def generator_loss(D_G):
    loss = -tf.reduce_mean(tf.log(D_G + 1e-12))
    loss_his = tf.summary.scalar("generator_loss", loss)

    return loss

def image_summary(label, image_data, channel=1):
    reshap_data = tf.reshape(image_data, [-1, image_width, image_height, channel])
    tf.summary.image(label, reshap_data, batch_size * 2)


def accuracy(pred_image, z_image, y_image):
    image_summary("z_image",  z_image, channel=1)
    image_summary("pred_image", pred_image * 0.5 + 0.5, channel=3)
    image_summary("y_image", y_image * 0.5 + 0.5, channel=3)
    return 1. - (tf.reduce_mean(tf.abs(tf.image.sobel_edges(z_image) - tf.image.sobel_edges(pred_image))))

def discriminator_train(loss, index, param):
    return tf.train.AdamOptimizer(learning_rate=D_learning_rate, beta1=0.5).minimize(loss, global_step=index, var_list=param)

def generator_train(loss, index, param):
    return tf.train.AdamOptimizer(learning_rate=G_learning_rate, beta1=0.5).minimize(loss, global_step=index, var_list=param)

def get_image(train_path, image_size=30,target_size=(image_width, image_height)):
    rgb_imgs = []
    light_imgs = []
    for index in range(1, image_size):
        imgData = io.imread(os.path.join(train_path ,"%d.jpg" % index))
        imgData = trans.resize(imgData, target_size)
        light_img = np.max(imgData, axis=2).reshape(image_width, image_height, 1)
        rgb_imgs.append(imgData)
        light_imgs.append(light_img)
    rgb_imgs = np.array(rgb_imgs) 
    light_imgs = np.array(light_imgs)
    return np.array(rgb_imgs), np.array(light_imgs)


def get_batch(size=10):
    rgb_imgs, _ = get_image('train/data1')
    indices = np.arange(len(rgb_imgs))
    np.random.shuffle(indices)
    
    rgbs = []
    lights = []
    light_labels = []
    
    for index in range(size):
        data1 = rgb_imgs[indices[index]]
        data2 = rgb_imgs[indices[index + 1]]
        for c in range(3):
            value = np.random.uniform(0.3, 1.3)
            data1[:, :, c] *= value
            data2[:, :, c] *= value
            if np.sum(data1[:, :, c] > 1.) > 0:
                data1[:, :, c] /= data1[:, :, c].max()
            if np.sum(data2[:, :, c] > 1.) > 0:
                data2[:, :, c] /= data2[:, :, c].max()
                
        light_img = np.max(data2, axis=2).reshape(image_width, image_height, 1)
        rgbs.append(data1)
        lights.append(light_img)
        light_labels.append(data2)
        
    return np.array(light_labels), np.array(rgbs), np.array(lights)

def get_test_batch(size=10):
    test_rgb, test_light = get_image('train/data1', image_size=size)
    train_rgb, train_light = get_image('train/data2', image_size=size)
        
    return np.concatenate([train_rgb[:size], test_rgb[:size]]), np.concatenate([test_light[:size], train_light[:size]])


if __name__ == '__main__':
    # init
    input_x = tf.placeholder(tf.float32, shape=[None, image_width, image_height, channel], name="input_x")
    input_y = tf.placeholder(tf.float32, shape=[None, image_width, image_height, channel], name="input_y")
    input_z = tf.placeholder(tf.float32, shape=[None, image_width, image_height, 1], name="input_z")

    # predict
    D_x_op = discriminator(input_x, input_y)
    G_z_op = generator(input_z, input_y)
    D_G_op = discriminator(G_z_op, input_y, tf.AUTO_REUSE)

    # loss
    discriminator_loss_op = discriminator_loss(D_x_op, D_G_op)
    generator_loss_op = generator_loss(D_G_op)

    # train
    D_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    G_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    discriminator_index = tf.Variable(0, name="discriminator_train_time")
    discriminator_train_op = discriminator_train(discriminator_loss_op, discriminator_index, D_param)

    generator_index = tf.Variable(0, name="generator_train_time")
    generator_train_op = generator_train(generator_loss_op, generator_index, G_param)

    # accuracy
    accuracy_op = accuracy(G_z_op, input_z, input_y)

    # graph

    summary_op = tf.summary.merge_all()
    session = tf.Session()
    summary_writer = tf.summary.FileWriter("log/", graph=session.graph)

    init_value = tf.global_variables_initializer()
    session.run(init_value)

    saver = tf.train.Saver()


    test_y, test_z = get_test_batch(test_size)
    test_y = (np.array(test_y) - 0.5) / 0.5
    for time in range(train_times):

        minibatch_x, minibatch_y, sample_z = get_batch(batch_size)
        minibatch_x = (np.array(minibatch_x) - 0.5) / 0.5
        minibatch_y = (np.array(minibatch_y) - 0.5) / 0.5
        
        
        for k in range(1):
            session.run(discriminator_train_op, feed_dict={input_x: minibatch_x, input_z: sample_z, input_y: minibatch_y})

        session.run(generator_train_op, feed_dict={input_x: minibatch_x, input_z: sample_z, input_y: minibatch_y})


        if (time + 1) % train_step == 0:
            feed = {input_x: test_y, input_z: test_z, input_y: test_y}
            op_list = [accuracy_op, discriminator_loss_op, generator_loss_op]
            accuracy, D_loss, G_loss = session.run(op_list, feed_dict=feed)
            
            accuracy = session.run(accuracy_op, feed_dict=feed)
            summary_str = session.run(summary_op, feed_dict=feed)
            summary_writer.add_summary(summary_str, session.run(generator_index))
            print("train times:", time + 1,
                                " accuracy:", accuracy,
                                    " D_loss:", D_loss,
                                    " G_loss", G_loss)
