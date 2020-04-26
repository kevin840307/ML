import tensorflow as tf
import tensorflow_tools as tf_tools

class Discriminator():
    def __init__(self, channel_min, out_size, channel_rate=2, name="discriminator"):
        self.channel_min = channel_min
        self.out_size = out_size
        self.channel_rate = channel_rate
        self.name = name

    def build(self, input, times=3, reuse=False, normal=True):
        output = input
        with tf.variable_scope(self.name, reuse=reuse):
            out_channel = self.channel_min
            for index in range(times):
                output = tf_tools.conv2d(output, out_channel=out_channel, activation='relu', normal=normal, name='conv_{}'.format(index))
                output = tf_tools.max_pool(output, name='pool_{}'.format(index))
                out_channel = out_channel * self.channel_rate

            output = tf.layers.flatten(output)
            output = tf_tools.layer(output, out_size=self.out_size, name='output')

            self.var_list = tf.trainable_variables(scope=tf.get_variable_scope().name)
        return output
