import tensorflow as tf
import tensorflow_tools as tf_tool
import numpy as np
from module import Model
from collections import defaultdict

GAMMA = 0.9
def image_summary(label, image_data, image_width=16, image_height=16):
    reshap_data = tf.reshape(image_data, [-1, image_width, image_height, 1])
    tf.summary.image(label, reshap_data,  50)

class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

class Policy(Model):
    def __init__(self, scope_name, env, state_shape, n_actions):
        self.scope_name = scope_name
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.env = env

    def build(self, train_fn=tf_tool.adam_fn, lr=1e-5, show_img=False):
        with tf.variable_scope(self.scope_name):
            self.state = tf.placeholder(tf.float32, [None, None] + list(self.state_shape), "state")
            self.condition = tf.placeholder(tf.float32, [None, None] + list(self.state_shape), "conditional")
            self.action = tf.placeholder(tf.int32, [None, None], "action")
            self.last_action = tf.placeholder(tf.float32, [None, None], "last_action")
            self.value_ = tf.placeholder(tf.float32, [None, None, 1], "v_next")
            self.reward = tf.placeholder(tf.float32, [None, None], 'reward')

            s_shape = tf.shape(self.state)
            self.batch_size, self.max_time = s_shape[0], s_shape[1]
            state = tf.concat([self.state, self.condition], axis=-1)
            s_shape = list(self.state_shape)
            s_shape[-1] = int(state.get_shape()[-1])
            self.input = tf.reshape(state, [-1] + s_shape)

            self.bind_network()
            self.var_list = tf.trainable_variables(scope=tf.get_variable_scope().name)
            self.train_fn = train_fn(lr)
            self.build_optimization()
            
            if show_img:
                image_summary('predict', self.state[:,-1])
                image_summary('condition', self.condition[:,-1])

    def bind_network(self):
        state_enc = tf_tool.conv2d(self.input, out_channel=32, activation='leaky_relu', init='xavier', name="conv{}".format(0))
        action_enc = tf_tool.layer(tf.expand_dims(self.last_action, -1), activation='leaky_relu', init='xavier', out_size=32)
        action_enc = tf.reshape(action_enc, [-1, 1, 1, 32])
        add_enc = state_enc + action_enc

        for idx in range(int(3)):
            add_enc = tf_tool.conv2d(add_enc, out_channel=32, activation='leaky_relu', init='xavier', name="add_enc_conv1_{}".format(idx))

        for idx in range(int(8)):
            originel_add = add_enc
            add_enc = tf_tool.conv2d(add_enc, out_channel=32, activation='leaky_relu', init='xavier', name="add_res_conv1_{}".format(idx))
            add_enc = tf_tool.conv2d(add_enc, out_channel=32, activation='leaky_relu', init='xavier', name="add_res_conv2_{}".format(idx)) + originel_add

        add_enc = tf.reshape(add_enc, [self.batch_size, self.max_time, 32 * np.prod(self.state_shape)])
        add_enc = tf_tool.layer(add_enc, out_size=128, activation='leaky_relu', init='xavier', name="layer{}".format(1))

        lstm_in = add_enc

        lstm = tf.nn.rnn_cell.BasicLSTMCell(512)
        def make_init(batch_size):
            c_init = np.zeros((batch_size, lstm.state_size.c), np.float32)
            h_init = np.zeros((batch_size, lstm.state_size.h), np.float32)
            return [c_init, h_init]

        self.state_init = keydefaultdict(make_init)
        c_in = tf.placeholder(tf.float32, [None, lstm.state_size.c], name="lstm_c_in")
        h_in = tf.placeholder(tf.float32, [None, lstm.state_size.h], name="lstm_h_in")
        self.init_state = [c_in, h_in]
        state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
        outputs, states = tf.nn.dynamic_rnn(lstm, lstm_in, initial_state=state_in, time_major=False)
        self.lstm_c, self.lstm_h = states

        with tf.variable_scope('Actor'):
            self.acts = tf_tool.layer(tf.nn.relu(outputs), out_size=self.n_actions, init='xavier', name='acts_prob')
            self.acts_prob = tf.nn.softmax(self.acts)
            self.acts_log_prob = tf.nn.log_softmax(self.acts)

        with tf.variable_scope('Critic'):
            self.value = tf_tool.layer(outputs, out_size=1, name='value')

    def build_optimization(self):
        adv = tf.reshape(GAMMA * self.value_ - self.value, [self.batch_size, self.max_time])
        self.td_error = self.reward + adv
        self.td_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.td_error), axis=1))

        action = tf.one_hot(tf.reshape(self.action, [-1, 1]), np.prod(self.n_actions))
        action = tf.reshape(action, [self.batch_size, self.max_time, self.n_actions])
        prob = tf.reduce_sum(self.acts_log_prob * action, -1)  # [2,0]
        entropy = tf.reduce_mean(tf.reduce_sum(self.acts_prob * self.acts_log_prob, axis=1))
        self.exp_v = -tf.reduce_mean(tf.reduce_sum(prob * tf.stop_gradient(self.td_error), axis=1))

        self.loss = self.td_loss + self.exp_v + 0.5 * entropy

        self.index = tf.Variable(0)
        self.train_op = self.train_fn.minimize(self.loss, global_step=self.index)

    def train(self, state, reward, condition, action, last_action, value):
        lstm_state_c, lstm_state_h = self.get_initial_features(len(state))
        value_ = value[:,1:]
        session = tf.get_default_session()
        feed_dict = {self.last_action:last_action, self.state: state, self.action: action, self.value_: value_, self.reward: reward, self.condition: condition, self.init_state[0]:lstm_state_c, self.init_state[1]:lstm_state_h}
        session.run([self.train_op], feed_dict=feed_dict)

    def predict_detail(self, state, condition, chooce_max=False, session=None, env=None):
        if env == None:
            env = self.env

        lstm_state_c, lstm_state_h = self.get_initial_features(1)
        last_action = -1
        datas = {'states':[],
                 'actions':[],
                 'last_actions':[],
                 'states_':[],
                 'rewards':[],
                 'conditions': [],
                 'values': [],
                 'reward_total':0}

        for k in range(env.episode_length):

            action, value, lstm_c, lstm_h = self.choose_action([[state]],[[condition]], lstm_state_c, lstm_state_h, [[last_action]], chooce_max, session)
                
            state_, reward, done, info = env.step([action])
            datas['states'].append(state)
            datas['last_actions'].append(last_action)
            datas['actions'].append(action)
            datas['conditions'].append(condition)
            datas['states_'].append(state_)
            datas['rewards'].append(reward)
            datas['values'].append(value)
            datas['reward_total'] += reward
            lstm_state_c = lstm_c
            lstm_state_h = lstm_h
            state = state_
            last_action = action
        datas['values'].append([0])
        return datas

    def choose_action(self, state, condition, lstm_c, lstm_h, last_action, chooce_max=False, session=None):
        if session == None:
            session = tf.get_default_session()
        feed_dict = {self.last_action:last_action, self.state: state, self.condition: condition, self.init_state[0]:lstm_c, self.init_state[1]:lstm_h}
        probs, lstm_c, lstm_h, value = session.run([self.acts_prob, self.lstm_c, self.lstm_h, self.value], feed_dict=feed_dict)   # get probabilities for all actions

        #if not chooce_max:
        if np.random.uniform() >= 0.7:
            action = np.random.choice(np.arange(probs.shape[-1]), p=probs.ravel())
        else:
            action = np.argmax(probs.ravel())

        return action, [np.squeeze(value)], lstm_c, lstm_h

    def get_initial_features(self, batch_size, flat=False):
        out = self.state_init[batch_size]
        if flat:
            out = [out[0][0], out[1][0]]
        return out