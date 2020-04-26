import tensorflow as tf
import numpy as np
from PIL import Image
import math

def adam_fn(lr=1e-4, beta1=0.9, beta2=0.999):
    return tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)

def sgd_fn(lr=1e-2):
    return tf.train.GradientDescentOptimizer(learning_rate=lr)

def moment_fn(lr=1e-2, momentum=0.9):
    return tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum)

def rmsprop_fn(lr=1e-2, decay=0.9):
    return tf.train.RMSPropOptimizer(learning_rate=lr, decay=decay)

def dice_loss(pred, label, smooth=1):
    intersection = tf.reduce_sum(pred * label, axis=0)
    union = tf.reduce_sum(pred, axis=0) + tf.reduce_sum(label, axis=0)
    return 1 - tf.reduce_mean((2. * intersection + smooth) / (union + smooth))

def focal_loss(pred, label, gamma=2., alpha=.25):
    class1_loss = alpha * label * tf.pow(1 - pred, gamma) * tf.log(pred + 1e-12)
    class2_loss = (1 - alpha) * (1 - label) * tf.pow(pred, gamma) * tf.log((1 - pred) + 1e-12)
    loss = -tf.reduce_mean(class1_loss + class2_loss)
    return loss

def cross_entropy_loss(pred, label):
    return -tf.reduce_mean(label * tf.log(pred + 1e-12) + (1. - label) * tf.log(1. - pred + 1e-12))

def sparse_softmax_cross_entropy_with_logits(logit, label):
    label = tf.cast(label, tf.int32)
    cross = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(label, squeeze_dims=[1]), logits=logit)
    loss = tf.reduce_mean(cross)
    return loss

def get_sync_op(from_list, to_list):
    assert len(from_list) == len(to_list), \
            "length of to variables should be same ({len(from_list)} != {len(to_list)})"
    syncs = []
    for from_v, to_v in zip(from_list, to_list):
        assert from_v.get_shape() == to_v.get_shape(), \
                "{from_v.get_shape()} != {to_v.get_shape()}" \
                " ({from_v.name}, {to_v.name})"
        sync = to_v.assign(from_v)
        syncs.append(sync)
    return tf.group(*syncs)

# batch
class BatchNormalization(object):
    def layer_batch_norm(self, x, is_train=tf.constant(True, dtype=tf.bool), decay=0.99, epsilon=1e-3, is_scale=True, is_center=True, name=''):
        size = x.get_shape()[-1]
        beta = None
        gamma = None
        if is_center:
            beta = tf.get_variable(name + "beta", [size], initializer=tf.zeros_initializer())
        if is_scale:
            gamma = tf.get_variable(name + "gamma", [size], initializer=tf.ones_initializer())

        batch_mean, batch_var = tf.nn.moments(x, [0], name=name + 'moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay, name=name + 'batch_exp_m')
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(is_train, mean_var_with_update, lambda:(ema_mean, ema_var))

        x_r = tf.reshape(x, [-1, 1, 1, size])
        #normed = tf.nn.batch_norm_with_global_normalization(x_r, mean, var,
        #beta,
        #gamma, 1e-8, True)
        normed = tf.nn.batch_normalization(x_r, mean, var, beta, gamma, epsilon)

        return tf.reshape(normed, [-1, size])

    def conv_batch_norm(self, x, is_train=tf.constant(True, dtype=tf.bool), decay=0.99, epsilon=1e-3, is_scale=True, is_center=True, name=''):
        size = x.get_shape()[-1]
        beta = None
        gamma = None
        if is_center:
            beta = tf.get_variable(name + "beta", [size], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
        if is_scale:
            gamma = tf.get_variable(name + "gamma", [size], initializer=tf.constant_initializer(value=1.0, dtype=tf.float32))

    
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name=name + 'moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay, name=name + 'batch_exp_m')
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_train, mean_var_with_update, lambda:(ema_mean, ema_var))
        #normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
        #beta, gamma, epsilon, False)
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)

        #mean_hist = tf.summary.histogram("meanHistogram", mean)
        #var_hist = tf.summary.histogram("varHistogram", var)


        return normed

    def batch_norm(self, x, is_train=tf.constant(True, dtype=tf.bool), decay=0.99, epsilon=1e-3, is_scale=True, is_center=True, name=''):
        return tf.layers.batch_normalization(x, training=is_train)
        #if len(x.get_shape()) == 2:
        #    return self.layer_batch_norm(x, is_train, decay, epsilon, is_scale, is_center, name)
        #else:
        #    return self.conv_batch_norm(x, is_train, decay, epsilon, is_scale, is_center, name)

# layer
def layer(input, weights_shape=None, weights=None, biases=None, out_size=1, normal=False, is_train=tf.constant(True, dtype=tf.bool), activation=None, init='he', name=''):
    if weights_shape == None:
        weights_shape = [input.get_shape().as_list()[-1], out_size]
        
    weights_param = 2.
    if init == 'xavier':
        weights_param = 1.

    size = weights_shape[0]
    if len(weights_shape) == 4:
        size = weights_shape[0] * weights_shape[1] * weights_shape[2]
    init = tf.random_normal_initializer(stddev=np.sqrt(weights_param / size))

    if weights == None:
        weights = tf.get_variable(name=name + "weights", shape=weights_shape, initializer=init)
        weights = tf.reshape(weights, [-1, weights_shape[-1]])

    if biases == None:
        biases = tf.get_variable(name=name + "biases", shape=weights_shape[-1], initializer=init)

    if len(input.get_shape()) == 3:
        weights = tf.tile(weights, [tf.shape(input)[0], 1])
        weights = tf.reshape(weights, [tf.shape(input)[0], input.get_shape().as_list()[-1], out_size])
        #weights = tf.tile(tf.expand_dims(weights, 0), [tf.shape(input)[0], 1, 1])

    out_put = tf.matmul(input, weights) + biases

    normal_class = BatchNormalization() 
    if normal:
       out_put = normal_class.batch_norm(out_put, is_train=is_train, name=name)

    activation_class = Activation()
    if activation != None and hasattr(activation_class, activation):
       func = getattr(activation_class, activation)
       out_put = func(out_put)

    return out_put

def conv2d(input, weights_shape=None, weights=None, biases=None, is_train=tf.constant(True, dtype=tf.bool), kernel_size=3, kernel_sizes=None, out_channel=1, strides=[1, 1, 1, 1], pad='SAME', normal=False, activation=None, init='he', bias_add=True, name=''):
    if weights_shape == None:
        if kernel_sizes == None:
            weights_shape = [kernel_size, kernel_size, input.get_shape().as_list()[-1], out_channel]
        else:
            weights_shape = kernel_sizes + [input.get_shape().as_list()[-1], out_channel]

    weights_param = 2.
    if init == 'xavier':
        weights_param = 1.

    size = weights_shape[0] * weights_shape[1] * weights_shape[2]
    weights_init = tf.random_normal_initializer(stddev=np.sqrt(weights_param / size))
    if weights == None:
        weights = tf.get_variable(name=name + "weights", shape=weights_shape, initializer=weights_init)
    out_put = tf.nn.conv2d(input, weights, strides=strides, padding=pad, name=name + "conv2d")

    if bias_add:
        if biases == None:
            biases = tf.get_variable(name=name + "biases", shape=weights_shape[3], initializer=tf.zeros_initializer())
        out_put = tf.nn.bias_add(out_put, biases)

    normal_class = BatchNormalization() 
    if normal:
       out_put = normal_class.batch_norm(out_put, is_train=is_train, name=name)

    activation_class = Activation()
    if activation != None and hasattr(activation_class, activation):
       func = getattr(activation_class, activation)
       out_put = func(out_put)

    return out_put

def atrous_conv2d(input, rate=1,weights_shape=None, is_train=tf.constant(True, dtype=tf.bool),kernel_size=3, out_channel=1, strides=[1, 1, 1, 1], pad='SAME', normal=False, activation=None, init='he', bias_add=True, name=''):
    if weights_shape == None:
        weights_shape = [kernel_size, kernel_size, input.get_shape().as_list()[-1], out_channel]

    weights_param = 2.
    if init == 'xavier':
        weights_param = 1.

    size = weights_shape[0] * weights_shape[1] * weights_shape[2]
    weights_init = tf.random_normal_initializer(stddev=np.sqrt(weights_param / size))
    weights = tf.get_variable(name=name + "weights", shape=weights_shape, initializer=weights_init)

    out_put = tf.nn.atrous_conv2d(input, weights, rate, padding=pad, name=name + "atrous_conv2d")
    if bias_add:
        biases = tf.get_variable(name=name + "biases", shape=weights_shape[3], initializer=tf.zeros_initializer())
        out_put = tf.nn.bias_add(out_put, biases)

    normal_class = BatchNormalization() 
    if normal:
       out_put = normal_class.batch_norm(out_put, is_train=is_train, name=name)

    activation_class = Activation()
    if activation != None and hasattr(activation_class, activation):
       func = getattr(activation_class, activation)
       out_put = func(out_put)

    return out_put

def spatial_pyramid_pool(previous_conv, previous_conv_size, out_pool_size, name=''):
    """
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
  
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    """
    for i in range(len(out_pool_size)):
        h_strd = h_size = math.ceil(float(previous_conv_size[0]) / out_pool_size[i])
        w_strd = w_size = math.ceil(float(previous_conv_size[1]) / out_pool_size[i])
        pad_h = int(out_pool_size[i] * h_size - previous_conv_size[0])
        pad_w = int(out_pool_size[i] * w_size - previous_conv_size[1])
        new_previous_conv = tf.pad(previous_conv, tf.constant([[0, 0], [0, pad_h], [0, pad_w], [0, 0]]), name=name+'_pad_{}'.format(i))
        max_pool = tf.nn.max_pool(new_previous_conv,
                        ksize=[1,h_size, h_size, 1],
                        strides=[1,h_strd, w_strd,1],
                        padding='SAME', name=name+'_max_pool_{}'.format(i))
        grid_size = out_pool_size[i] ** 2
        if (i == 0):
            spp = tf.reshape(max_pool, [-1, grid_size * previous_conv.get_shape()[-1]], name=name+'_spp_{}'.format(i))
        else:
            spp = tf.concat(axis=1, values=[spp, tf.reshape(max_pool, [-1, grid_size * previous_conv.get_shape()[-1]])], name=name+'_spp_{}'.format(i))
  
    return spp

def lstm(input, hidden_dim, keep_prob=None):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
        if keep_prob != None:
            lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        # stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([dropout_lstm] * 2,
        # state_is_tuple=True)
        lstm_outputs, state = tf.nn.dynamic_rnn(lstm, input, dtype=tf.float32)
        #return tf.squeeze(tf.slice(lstm_outputs, [0,
        #tf.shape(lstm_outputs)[1]-1, 0], [tf.shape(lstm_outputs)[0], 1,
        #tf.shape(lstm_outputs)[2]]))
        return lstm_outputs, state

def upsampling(input, size):
    return tf.image.resize_images(input, size)

def drop(input, prob):
    return tf.nn.dropout(input, prob)

def conv2d_transpose(input, out_channel, output_size=None, is_train=tf.constant(True, dtype=tf.bool), kernel_size=3, strides=[1, 2, 2, 1], pad='SAME', normal=False, activation=None, init='he', bias_add=True, name=''):
    #conv2d_tran = tf.layers.conv2d_transpose(input, filters=filters,
    #kernel_size=kernel_size, strides=2, padding='SAME')
    input_shape = input.get_shape().as_list()
    if output_size == None:
        output_shape = [tf.shape(input)[0], input_shape[1] * 2, input_shape[2] * 2, out_channel]
    else:
        output_shape = [tf.shape(input)[0]] + output_size + [out_channel]

    weights_param = 2.
    if init == 'xavier':
        weights_param = 1.

    weights_init = tf.random_normal_initializer(stddev=np.sqrt(weights_param / (kernel_size * kernel_size * out_channel)))
    weights = tf.get_variable(name + 'transpose_weights', [kernel_size, kernel_size, out_channel, input_shape[3]], initializer=weights_init)

    out_put = tf.nn.conv2d_transpose(input, weights, output_shape=output_shape, strides=strides, padding=pad)

    if bias_add:
        biases = tf.get_variable(name + 'transpose_biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        out_put = tf.nn.bias_add(out_put, biases)

    normal_class = BatchNormalization() 
    if normal:
       out_put = normal_class.batch_norm(out_put, is_train=is_train, name=name)

    activation_class = Activation()
    if activation != None and hasattr(activation_class, activation):
       func = getattr(activation_class, activation)
       out_put = func(out_put)
    return out_put


# sampleing
def max_pool(input, ksize=2, k=2, pad='SAME', name=''):
    return tf.nn.max_pool(input, ksize = [1, ksize, ksize, 1], strides = [1, k, k, 1], padding = pad, name='pool' + name)


# activation funtion
class Activation(object):
    def selu(self, x):
        return tf.nn.selu(x)
        #alpha = 1.6732632423543772848170429916717
        #scale = 1.0507009873554804934193349852946
        #return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)

    def relu(self, x):
        return tf.nn.relu(x)

    def sigmoid(self, x):
        return tf.nn.sigmoid(x)

    def softplus(self, x):
        return tf.nn.softplus(x)

    def tanh(self, x):
        return tf.nn.tanh(x)

    def leaky_relu(self, x):
        return tf.nn.leaky_relu(x)

    def softmax(self, x):
        return tf.nn.softmax(x)

#image
def resize(x, original_width, original_height, width, height):
    data = []
    for index in range(len(x)):
        img = Image.fromarray(x[index].reshape([original_width, original_height])).resize((width, height))
        data.append(np.array(img.getdata()))
    return np.array(data)