import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
from PIL import Image
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 50
train_times = 1
train_step = 10
height = 28
width = 28
channels = 1
num_label = 10

# [kernel height, filter weight, input channel, out channel]
conv1_size = [9, 9, 1, 256]
caps1_size = [9, 9, 256, 32 * 8]

def conv2d(input, weight_shape, strides=[1, 1, 1, 1]):
    size = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weights_init = tf.random_normal_initializer(stddev=np.sqrt(2. / size))
    biases_init = tf.zeros_initializer()
    weights = tf.get_variable(name="weights", shape=weight_shape, initializer=weights_init)
    biases = tf.get_variable(name="biases", shape=weight_shape[3], initializer=biases_init)

    conv_out = tf.nn.conv2d(input, weights, strides=strides, padding='VALID')
    conv_add = tf.nn.bias_add(conv_out, biases)
    #conv_batch = conv_batch_norm(conv_add, weight_shape[3], tf.constant(True,
    #dtype=tf.bool))
    #output = tf.nn.relu(conv_batch)
    output = tf.nn.relu(conv_add)

    return output

class CapsLayer(object):
    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
            self.num_outputs = num_outputs
            self.vec_len = vec_len
            self.with_routing = with_routing
            self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride
            if not self.with_routing:
                # input: [batch_size, 20, 20, 256]
                # conv kernel [filter weight, filter height, output channel, input channel]
                # output: [batch_size, 6, 6, 32 * 8]
                input_shape = input.get_shape()
                caps_size = [kernel_size, kernel_size, input_shape[3].value, self.num_outputs * self.vec_len]
                capsules  = conv2d(input, caps_size, [1, stride, stride, 1])
    
                # input: [batch_size, 6 * 6 * 32, 8, 256]
                # caps_child = ((input_shape[1].value - kernel_size + 1) / stride) * ((input_shape[2].value - kernel_size + 1) / stride) * self.num_outputs
                # output[batch_size, 1152, 8, 1]
                caps_child = int((input_shape[1].value - kernel_size + 1) / stride) ** 2 * self.num_outputs
                capsules = tf.reshape(capsules, (-1, caps_child, self.vec_len, 1))
                capsules = squash(capsules)

                return capsules
        if self.layer_type == 'FC':
            if self.with_routing:
                self.input = tf.reshape(input, shape=(-1, input.shape[1].value, 1, input.shape[2].value, 1))

                with tf.variable_scope("routing"):
                    capsules = routing(self.input, num_outputs=num_label, num_dims=self.vec_len)
                    capsules = tf.squeeze(capsules, axis=1)
                return capsules
# norm        
def squash(s, epsilon=1e-7):
    vec_squared_norm = tf.reduce_sum(tf.square(s), -1, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * s  # element-wise
    return(vec_squashed)

def routing(input, num_outputs=10, num_dims=16):
    input_shape = input.get_shape()
    
    weights_shape = [1, input_shape[1], num_dims * num_outputs, input_shape[-2], input_shape[-1]]
    weights_init = tf.random_normal_initializer(stddev=0.01)
    W = tf.get_variable('Weight', shape=weights_shape, dtype=tf.float32, initializer=weights_init)
    biases = tf.get_variable('bias', shape=(1, 1, num_outputs, num_dims, 1))
    
    input = tf.tile(input, [1, 1, num_dims * num_outputs, 1, 1])

    u_hat = tf.reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, input_shape[1], num_outputs, num_dims, 1])
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
    b = tf.zeros_like(u_hat, dtype=tf.float32)
    for r_iter in range(3):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b, axis=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == 3 - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                # assert s_J.get_shape() == [cfg.batch_size, 1, num_outputs, num_dims, 1]

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                # assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
            elif r_iter < 3 - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, input_shape[1], 1, 1, 1])
                u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                # assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b += u_produce_v

    return(v_J)


    




def predict(x, y, training):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    def rot():
        return tf.image.rot90(x)
    
    x = tf.cond(training, rot, lambda:(x))
    
    with tf.variable_scope("conv1_scope"):
        conv1 = conv2d(x, conv1_size)
    with tf.variable_scope('PrimaryCaps_layer'):
        primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
        caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
    with tf.variable_scope('DigitCaps_layer'):
        digitCaps = CapsLayer(num_outputs=num_label, vec_len=16, with_routing=True, layer_type='FC')
        caps2 = digitCaps(caps1)
    with tf.variable_scope('Encoder'):
        masked_v = tf.multiply(tf.squeeze(caps2), tf.reshape(y, (-1, num_label, 1)))
        
    with tf.variable_scope('Decoder'):
        vector_j = tf.reshape(masked_v, shape=(-1, 16 * num_label))
        fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
        fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
        decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=height * width * channels, activation_fn=tf.sigmoid)
    return (caps2, decoded)
    
#def masking(caps, y):
#    with tf.variable_scope('Encoder'):
#        masked_v = tf.multiply(tf.squeeze(caps), tf.reshape(y, (-1, num_label, 1)))
#    with tf.variable_scope('Decoder'):
#        vector_j = tf.reshape(masked_v, shape=(batch_size, -1))
#        fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
#        fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
#        decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=height * width * channels, activation_fn=tf.sigmoid)
#    return decoded


m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5
def loss(v, decoded, x, y):
    T = y
    print("v:", v.shape)
    #v_norm = tf.norm(v, axis=-2, keep_dims=True, name="caps2_output_norm")
    v_norm = tf.sqrt(tf.reduce_sum(tf.square(v), axis=2, keepdims=True) + 1e-8)
    print("v_norm:", v_norm.shape)
    FP_raw = tf.square(tf.maximum(0., m_plus - v_norm), name="FP_raw")
    FP = tf.reshape(FP_raw, shape=(-1, 10), name="FP")
    print("FP:", FP.shape)
    FN_raw = tf.square(tf.maximum(0., v_norm - m_minus), name="FN_raw")
    FN = tf.reshape(FN_raw, shape=(-1, 10), name="FN")
    print("FN:", FN.shape)
    L = tf.add(T * FP, lambda_ * (1.0 - T) * FN, name="L")
    print("L:", L.shape)
    margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
    print("margin_loss:", margin_loss.shape)

     # 2. The reconstruction loss
    orgin = tf.reshape(x, shape=(-1, height * width * channels))
    squared = tf.square(decoded - orgin)
    reconstruction_err = tf.reduce_mean(squared)
    
    total_loss = margin_loss + 0.005 * reconstruction_err
    return total_loss

def train(loss, index):
    return tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss, global_step=index)

def accuracy(output, t):
    comparison = tf.equal(tf.reshape(output, [-1]), tf.argmax(t, 1))
    y = tf.reduce_mean(tf.cast(comparison, tf.float32))
    
    return y

mnist = input_data.read_data_sets("MNIST/", one_hot=True)
input_x = tf.placeholder(tf.float32, shape=[None, 784], name="input_x")
input_y = tf.placeholder(tf.float32, shape=[None, 10], name="input_y")
training =tf.placeholder(tf.bool, name="training")


(predict_op, decoded_op) = predict(input_x, input_y, training)
#predict_norm_op = tf.norm(predict_op, axis=-2, keep_dims=True)
predict_norm_op = tf.sqrt(tf.reduce_sum(tf.square(predict_op), axis=2, keepdims=True) + 1e-8)
predict_softmax_op = tf.nn.softmax(predict_norm_op, axis=1)
predict_max_op = tf.argmax(predict_softmax_op, axis=1)
predict_out_op = tf.reshape(predict_max_op, shape=(-1, ))

loss_op = loss(predict_op, decoded_op, input_x, input_y)

# train
index = tf.Variable(0, name="train_time")
train_op = train(loss_op, index)

# accuracy
accuracy_op = accuracy(predict_out_op, input_y)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
init_value = tf.global_variables_initializer()
session.run(init_value)

warp_img = []
for index in range(len(mnist.validation.images)):
    img = Image.fromarray(mnist.validation.images[index].reshape([28, 28])).rotate(45)
    warp_img.append(np.array(img.getdata()))
warp_img = np.array(warp_img)

for time in range(train_times):
    avg_loss = 0.
    total_batch = int(mnist.train.num_examples / batch_size)
    validation_batch = int(len(mnist.validation.images) / batch_size)

    for i in range(total_batch):
        minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
        session.run(train_op, feed_dict={input_x: minibatch_x, input_y: minibatch_y, training: False})
        avg_loss += session.run(loss_op, feed_dict={input_x: minibatch_x, input_y: minibatch_y, training: False}) / 10
        if ((time * total_batch) + i) % 10 == 0:
            accuracy = session.run(accuracy_op, feed_dict={input_x: mnist.validation.images[0:batch_size], input_y: mnist.validation.labels[0:batch_size], training: False})
            warp_accuracy = session.run(accuracy_op, feed_dict={input_x: warp_img[0:batch_size], input_y: mnist.validation.labels[0:batch_size], training: False})
            #p = session.run(tf.argmax(predict_norm_op, 1), feed_dict={input_x: mnist.validation.images[0:batch_size]})
            #print(p.reshape(-1))
            
            print(" avg_loss:", avg_loss,
                    " accuracy:", accuracy,
                    " warp_accuracy:", warp_accuracy)
            avg_loss = 0
    for index in range(validation_batch):
        accuracy += session.run(accuracy_op, feed_dict={input_x: mnist.validation.images[index * batch_size:(index + 1) * batch_size], input_y: mnist.validation.labels[index * batch_size:(index + 1) * batch_size], training: False})
        warp_accuracy += session.run(accuracy_op, feed_dict={input_x: warp_img[index * batch_size:(index + 1) * batch_size], input_y: mnist.validation.labels[index * batch_size:(index + 1) * batch_size], training: False})
    print(" avg_loss:", avg_loss,
        " accuracy:", accuracy / validation_batch,
        " warp_accuracy:", warp_accuracy / validation_batch)
session.close()

