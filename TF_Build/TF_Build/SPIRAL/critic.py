import tensorflow as tf
import tensorflow_tools as tf_tool
import numpy as np

GAMMA = 0.9
def image_summary(label, image_data, image_width=16, image_height=16):
    reshap_data = tf.reshape(image_data, [-1, image_width, image_height, 1])
    tf.summary.image(label, reshap_data,  50)

class Critic(object):
    def __init__(self, scope_name, env, states_shape, n_features):
        self.scope_name = scope_name
        self.states_shape = states_shape
        self.env = env
        

    def build(self, train_fn=tf_tool.adam_fn, lr=1e-5):
        with tf.variable_scope(self.scope_name):
            self.state = tf.placeholder(tf.float32, [None, None] + list(self.states_shape), "state")
            self.condition = tf.placeholder(tf.float32, [None, None] + list(self.states_shape), "conditional")

            self.value_ = tf.placeholder(tf.float32, [None, None, 1], "value_next")
            self.reward = tf.placeholder(tf.float32, [None, None], 'reward')

            x_shape = tf.shape(self.state)
            self.batch_size, self.max_time = x_shape[0], x_shape[1]

            state = tf.concat([self.state, self.condition], axis=-1)
            x_shape = list(self.states_shape)
            x_shape[-1] = int(state.get_shape()[-1])
            self.input = tf.reshape(state, [-1] + x_shape)

            self.build_network()
            self.train_fn = train_fn(lr)
            self.build_optimization()
        image_summary('predict', self.state)
        image_summary('condition', self.condition)

    def build_network(self):
        x_enc = tf_tool.conv2d(self.input, activation='leaky_relu', out_channel=32, name="conv{}".format(0))

        for idx in range(int(5)):
            originel_add = x_enc
            x_enc = tf_tool.conv2d(x_enc, out_channel=32, activation='leaky_relu', init='xavier', name="add_res_conv1_{}".format(idx))
            x_enc = tf_tool.conv2d(x_enc, out_channel=32, activation='leaky_relu', init='xavier', name="add_res_conv2_{}".format(idx)) + originel_add

        add_enc = tf.reshape(x_enc, [self.batch_size, self.max_time, 32 * np.prod(self.states_shape)])
        add_enc = tf_tool.layer(add_enc, out_size=256, activation='leaky_relu', init='xavier', name="layer{}".format(1))
        lstm_in = add_enc

        lstm = tf.nn.rnn_cell.BasicLSTMCell(256, name='lstm', state_is_tuple=True)
        initial_state = lstm.zero_state(self.batch_size, dtype=tf.float32)
        outputs, states = tf.nn.dynamic_rnn(lstm, lstm_in, initial_state=initial_state, time_major=False)

        self.value = tf_tool.layer(outputs, out_size=1, name='value')

    def build_optimization(self):
        diff = tf.reshape(GAMMA * self.value_ - self.value, [self.batch_size, self.max_time])
        self.td_error = self.reward + diff
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.td_error), axis=1))

        self.index = tf.Variable(0)
        self.train_op = self.train_fn.minimize(self.loss, global_step=self.index)
        
    def train(self, state, reward, condition):
        session = tf.get_default_session()
        value_ = session.run(self.value, {self.state: state, self.condition: condition})
        value_ = np.concatenate([value_, np.zeros(shape=(len(state), 1, 1))], axis=1)
        value_ = value_[:,1:]
        feed_dict = {self.state: state, self.value_: value_, self.reward: reward, self.condition: condition}
        td_error, _ = session.run([self.td_error, self.train_op], feed_dict=feed_dict)

        return td_error
