import tensorflow as tf
import tensorflow_tools as tf_tools

class Generator():
    def __init__(self, channel_min, out_size, channel_rate=2, name="generator"):
        assert len(out_size) == 3

        self.channel_min = channel_min
        self.out_size = out_size
        self.channel_rate = channel_rate
        self.name = name

    def build(self, input, times=3):
        with tf.variable_scope(self.name):
            size = self.out_size[0] * self.out_size[1]
            output = tf_tools.layer(input, out_size=size, activation='relu', normal=True, name='layer1')
            output = tf.reshape(output, shape=[-1, self.out_size[0], self.out_size[1], 1])

            out_channel = self.channel_min
            for index in range(times):
                output = tf_tools.conv2d(output, out_channel=out_channel, activation='relu', normal=True, name='conv_{}'.format(index))
                out_channel = out_channel * self.channel_rate

            output = tf_tools.conv2d(output, out_channel=self.out_size[-1], name='output')

            self.var_list = tf.trainable_variables(scope=tf.get_variable_scope().name)
        return output
