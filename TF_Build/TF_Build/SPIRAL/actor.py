import tensorflow as tf
import tensorflow_tools as tf_tool
import numpy as np

from collections import defaultdict
class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

class Actor(object):
    def __init__(self, scope_name, env, states_shape, n_actions):
        self.scope_name = scope_name
        self.states_shape = states_shape
        self.env = env
        self.n_actions = n_actions


    def build(self, train_fn=tf_tool.adam_fn, lr=1e-5):
        with tf.variable_scope(self.scope_name):
            self.state = tf.placeholder(tf.float32, [None, None] + list(self.states_shape), "state")
            self.condition = tf.placeholder(tf.float32, [None, None] + list(self.states_shape), "conditional")

            self.action = tf.placeholder(tf.int32, [None, None], "action")
            self.td_error = tf.placeholder(tf.float32, [None, None], "td_error")

            x_shape = tf.shape(self.state)
            self.batch_size,self. max_time = x_shape[0], x_shape[1]
            state = tf.concat([self.state, self.condition], axis=-1)
            x_shape = list(self.states_shape)
            x_shape[-1] = int(state.get_shape()[-1])
            self.input = tf.reshape(state, [-1] + x_shape)

            self.value = self.build_network()
            self.train_fn = train_fn(lr)
            self.build_optimization()

    def build_network(self):
        x_enc = tf_tool.conv2d(self.input, out_channel=32, activation='leaky_relu', init='xavier', name="conv{}".format(0))

        for idx in range(int(5)):
            originel_add = x_enc
            x_enc = tf_tool.conv2d(x_enc, out_channel=32, activation='leaky_relu', init='xavier', name="add_res_conv1_{}".format(idx))
            x_enc = tf_tool.conv2d(x_enc, out_channel=32, activation='leaky_relu', init='xavier', name="add_res_conv2_{}".format(idx)) + originel_add
            
        add_enc = tf.reshape(x_enc, [self.batch_size, self.max_time, 32 * np.prod(self.states_shape)])
        add_enc = tf_tool.layer(add_enc, out_size=256, activation='leaky_relu', init='xavier', name="layer{}".format(1))

        lstm_in = add_enc

        lstm = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)
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

        acts_logit = tf_tool.layer(outputs, out_size=self.n_actions, init='xavier', name='acts_prob') 
        self.acts_prob = tf.nn.softmax(acts_logit)
        self.acts_log_prob = tf.nn.log_softmax(acts_logit)

    def build_optimization(self):
        action = tf.one_hot(tf.reshape(self.action, [-1, 1]), np.prod(self.n_actions))
        action = tf.reshape(action, [self.batch_size, self.max_time, self.n_actions])

        log_prob = tf.reduce_sum(self.acts_log_prob * action, -1)  # [2,0]
        entropy = tf.reduce_mean(tf.reduce_sum(self.acts_prob * self.acts_log_prob, axis=1))
        self.exp_v = tf.reduce_mean(tf.reduce_sum(log_prob * self.td_error, axis=1))  # advantage (TD_error) guided loss

        self.loss = -self.exp_v + 0.5 * entropy

        self.index = tf.Variable(0)
        self.train_op = self.train_fn.minimize(self.loss, global_step=self.index)

    def predict_detail(self, state, condition, chooce_max=False):
        lstm_state_c, lstm_state_h = self.get_initial_features(1)
        last_action = -1
        datas = {'states':[],
                 'actions':[],
                 'last_actions':[],
                 'states_':[],
                 'rewards':[],
                 'conditions': [],
                 'reward_total':0}
        for k in range(self.env.episode_length):

            action, lstm_c, lstm_h = self.choose_action([[state]],[[condition]], lstm_state_c, lstm_state_h, chooce_max)
                

            state_, reward, done, info = self.env.step([action])
            datas['states'].append(state)
            datas['last_actions'].append(last_action)
            datas['actions'].append(action)
            datas['conditions'].append(condition)
            datas['states_'].append(state_)
            datas['rewards'].append(reward)
            datas['reward_total'] += reward
            lstm_state_c = lstm_c
            lstm_state_h = lstm_h
            state = state_
            last_action = action

        return datas

    def predict(self, state, condition, chooce_max=False):
        lstm_state_c, lstm_state_h = self.get_initial_features(1)

        for k in range(self.env.episode_length):

            action, lstm_c, lstm_h = self.choose_action([[state]],[[condition]], lstm_state_c, lstm_state_h, chooce_max)
                
            state_, reward, done, info = self.env.step([action])

            lstm_state_c = lstm_c
            lstm_state_h = lstm_h
            state = state_
        return state_, reward

    def train(self, state, action, td, condition):
        session = tf.get_default_session()
        lstm_state_c, lstm_state_h = self.get_initial_features(len(state))
        feed_dict = {self.state: state, self.action: action, self.td_error: td, self.condition: condition, self.init_state[0]:lstm_state_c, self.init_state[1]:lstm_state_h}
        _, exp_v = session.run([self.train_op, self.exp_v], feed_dict=feed_dict)
        return exp_v

    def choose_action(self, state, condition, lstm_c, lstm_h, chooce_max=False):
        session = tf.get_default_session()
        feed_dict = {self.state: state, self.condition: condition, self.init_state[0]:lstm_c, self.init_state[1]:lstm_h}
        probs, lstm_c, lstm_h = session.run([self.acts_prob, self.lstm_c, self.lstm_h], feed_dict=feed_dict)   # get probabilities for all actions

        if not chooce_max:
            action = np.random.choice(np.arange(probs.shape[-1]), p=probs.ravel())
        else:
            action = np.argmax(probs.ravel())

        return action, lstm_c, lstm_h

    def get_initial_features(self, batch_size, flat=False):
        out = self.state_init[batch_size]
        if flat:
            out = [out[0][0], out[1][0]]
        return out