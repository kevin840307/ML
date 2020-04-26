import sys, os
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import multiprocessing
import time
import threading
import numpy as np
import tensorflow as tf
import gym
import pandas as pd
from toy import Toy
import tensorflow_tools as tf_tool
import imageio
from module import Model
from collections import namedtuple
tc = tf.nn.rnn_cell
N_WORKERS = multiprocessing.cpu_count()
Batch = namedtuple("Batch", ["states", "actions", "last_actions", "states_", "rewards", "conditions", "values", "reward_total"])

def imsave1(path, img):
    img = np.clip(img * 127.5 + 127.5, 0, 255).astype(np.uint8)
    img = np.tile(img, [1, 1, 3])
    imageio.imwrite(path, img)

env = Toy()
N_F = env.observation_shape
N_A = 16
GLOBAL_POLICY_NET_SCOPE = 'Global_Policy'
BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
OUTPUT_GRAPH = False
MAX_EPISODE = 3000000
MAX_EP_STEPS = 20   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR = 1e-4    # learning rate for actor
GLOBAL_EP = 0
OPT = tf.train.AdamOptimizer(LR)

def image_summary(label, image_data, image_width=16, image_height=16):
    reshap_data = tf.reshape(image_data, [-1, image_width, image_height, 1])
    tf.summary.image(label, reshap_data,  TEST_BATCH_SIZE * 2 + env.batch_size * 2)

from collections import defaultdict
class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

class Policy(Model):
    def __init__(self, env, state_shape, n_actions, scope, globalP=None):
        self.env = env
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.state = tf.placeholder(tf.float32, [None, None] + list(self.state_shape), "state")
        self.condition = tf.placeholder(tf.float32, [None, None] + list(self.state_shape), "conditional")
        self.last_action = tf.placeholder(tf.float32, [None, None], "last_action")
        self.action = tf.placeholder(tf.int32, [None, None], "action")
        self.value_ = tf.placeholder(tf.float32, [None, None, 1], "v_next")
        self.reward = tf.placeholder(tf.float32, [None, None], 'reward')
        s_shape = tf.shape(self.state)
        self.batch_size, self.max_time = s_shape[0], s_shape[1]

        with tf.variable_scope(scope):
            self.acts, self.value = self.bind_model()
            self.acts_prob = tf.nn.softmax(self.acts)
            self.acts_log_prob = tf.nn.log_softmax(self.acts)
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        if GLOBAL_POLICY_NET_SCOPE != scope:
            value_ = tf.reshape(self.value_ , [self.batch_size, self.max_time])
            with tf.variable_scope(scope + '/td_loss'):
                adv = tf.reshape(GAMMA * self.value_ - self.value, [self.batch_size, self.max_time])
                self.td_error = self.reward + adv
                self.td_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.td_error), axis=1))
            with tf.variable_scope(scope + '_exp_v'):
                #prob = tf.gather(self.acts_log_prob, self.action, axis=-1)
                #action_hot = tf.one_hot(self.action, np.prod(N_A))
                #prob = tf.reduce_sum(action_hot * self.acts_log_prob, [-1])
                action = tf.one_hot(tf.reshape(self.action, [-1, 1]), np.prod(N_A))
                action = tf.reshape(action, [self.batch_size, self.max_time, N_A])
                prob = tf.reduce_sum(self.acts_log_prob * action, -1)  # [2,0]

                entropy = tf.reduce_mean(tf.reduce_sum(self.acts_prob * self.acts_log_prob, axis=1))
                self.exp_v = -tf.reduce_mean(tf.reduce_sum(prob * tf.stop_gradient(self.td_error), axis=1))
            with tf.variable_scope(scope + '_loss'):
                self.loss = self.td_loss + self.exp_v + 0.5 * entropy
                self.grads = tf.gradients(self.loss , self.var_list)
                grads = self.grads
            with tf.variable_scope(scope + '_sync'):
                with tf.variable_scope('pull'):
                    self.pull_var_list_op = [l_p.assign(g_p) for l_p, g_p in zip(self.var_list, globalP.var_list)]
                with tf.variable_scope('push'):
                    self.update_var_list_op = OPT.apply_gradients(zip(grads, globalP.var_list))
        else:
            image_summary('predict', self.state[:,-1])
            image_summary('condition', self.condition[:,-1])

    def bind_model(self):
        s_shape = tf.shape(self.state)
        batch_size, max_time = s_shape[0], s_shape[1]
        state = tf.concat([self.state, self.condition], axis=-1)
        #state = self.state
        s_shape = list(self.state_shape)
        s_shape[-1] = int(state.get_shape()[-1])
        state = tf.reshape(state, [-1] + s_shape)
        last_action = self.last_action

        with tf.variable_scope('Policy'):
            state_enc = tf_tool.conv2d(state, out_channel=32, activation='leaky_relu', init='xavier', name="conv{}".format(0))
            last_action_enc = tf_tool.layer(tf.expand_dims(last_action, -1), activation='leaky_relu', init='xavier', out_size=32)
            last_action_enc = tf.reshape(last_action_enc, [-1, 1, 1, 32])
            add_enc = state_enc + last_action_enc

            for idx in range(int(3)):
                add_enc = tf_tool.conv2d(add_enc, out_channel=32, activation='leaky_relu', init='xavier', name="add_enc_conv1_{}".format(idx))
                
            for idx in range(int(8)):
                originel_add = add_enc
                add_enc = tf_tool.conv2d(add_enc, out_channel=32, activation='leaky_relu', init='xavier', name="add_res_conv1_{}".format(idx))
                add_enc = tf_tool.conv2d(add_enc, out_channel=32, activation='leaky_relu', init='xavier', name="add_res_conv2_{}".format(idx)) + originel_add

            add_enc = tf.reshape(add_enc, [self.batch_size, self.max_time, 32 * np.prod(self.state_shape)])
            add_enc = tf_tool.layer(add_enc, out_size=256, activation='leaky_relu', init='xavier', name="layer{}".format(1))

            lstm_in = add_enc

            lstm = tf.nn.rnn_cell.BasicLSTMCell(512, name='lstm')
            def make_init(batch_size):
                c_init = np.zeros((batch_size, lstm.state_size.c), np.float32)
                h_init = np.zeros((batch_size, lstm.state_size.h), np.float32)
                return [c_init, h_init]

            self.state_init = keydefaultdict(make_init)
            c_in = tf.placeholder(tf.float32, [None, lstm.state_size.c], name="lstm_c_in")
            h_in = tf.placeholder(tf.float32, [None, lstm.state_size.h], name="lstm_h_in")
            self.init_state = [c_in, h_in]
            state_in = tc.LSTMStateTuple(c_in, h_in)
            outputs, self.states = tf.nn.dynamic_rnn(lstm, lstm_in, initial_state=state_in, time_major=False)
            self.lstm_c, self.lstm_h = self.states

            with tf.variable_scope('Actor'):
                acts = tf_tool.layer(tf.nn.relu(outputs), out_size=self.n_actions, init='xavier', name='acts_prob') 
            with tf.variable_scope('Critic'):
                value = tf_tool.layer(outputs, out_size=1, name='value')

        return acts, value

    def update_global(self, datas):  # run by a local
        lstm_state_c, lstm_state_h = self.get_initial_features(len(datas.states))
        value_ = datas.values[:,1:] 

        feed_dict = {self.reward:datas.rewards ,self.state: datas.states, self.action: datas.actions,self.last_action:datas.last_actions, self.value_: value_, self.condition: datas.conditions, self.init_state[0]:lstm_state_c, self.init_state[1]:lstm_state_h}
        _, loss = SESS.run([self.update_var_list_op, self.loss], feed_dict)  # local grads applies to global net
        return loss

    def pull_global(self):  # run by a local
        SESS.run([self.pull_var_list_op])

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
            action, value, lstm_c, lstm_h = self.choose_action([[state]],[[condition]], [[last_action]], lstm_state_c, lstm_state_h, chooce_max, session)
                
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

    def choose_action(self, state, condition, last_action, last_state_c, last_state_h, chooce_max=False, session=None):
        probs, lstm_c, lstm_h, value = SESS.run([self.acts_prob, self.lstm_c, self.lstm_h, self.value], {self.state: state, self.condition: condition, self.last_action: last_action, self.init_state[0]:last_state_c, self.init_state[1]:last_state_h})   # get probabilities for all actions
        if np.random.uniform() >= 0.7:
            action = np.random.choice(np.arange(probs.shape[-1]), p=probs.ravel())
        else:
            action = np.argmax(np.reshape(probs.ravel(), (-1)))
        return action, value,lstm_c, lstm_h

    def get_initial_features(self, batch_size, flat=False):
        out = self.state_init[batch_size]
        if flat:
            out = [out[0][0], out[1][0]]
        return out

class Worker(object):
    def __init__(self, name, globalP):
        self.env = Toy()
        self.name = name
        self.policy = Policy(env=self.env,state_shape=N_F, n_actions=N_A, globalP=globalP, scope=name + '/Policy')

    def work(self):
        train_times = 0

        while not COORD.should_stop() and train_times < MAX_EPISODE:
            #datas = get_data(self.policy, self.env, 3)
            datas = get_train_data(self.policy.env, self.policy)
            self.policy.pull_global()
            self.policy.update_global(datas)     # true_gradient = grad[logPi(s,a) * td_error]

            #if GLOBAL_EP % MAX_EP_STEPS == 0:
            #    print(self.name + ' time:', train_times,
            #            ' reward_total:', datas.reward_total,
            #            ' mean_reward:', datas.reward_total / BATCH_SIZE)
            train_times += 1

class SummerWorker(object):
    def __init__(self, globalP):
        self.policy = globalP

    def work(self):
        train_times = 0

        while not COORD.should_stop() and train_times < MAX_EPISODE:
            time.sleep(0.5)
            if train_times % 5 == 0:
                #datas, test_reward_total, train_reward_total = get_test_data(self.env, self.policy)
                #print(' time:', train_times,
                #        ' test_reward_total:', test_reward_total / TEST_BATCH_SIZE,
                #        ' train_reward_total:', train_reward_total / env.batch_size)
                #datas = get_data(self.policy, self.policy.env, 3)
                datas = get_train_data(self.policy.env, self.policy)
                print(' time:', train_times,
                        ' train_reward_total:', datas.reward_total)

                summary_str = SESS.run(SUMMARY_OP, feed_dict={self.policy.state: np.expand_dims(datas.states_[:, -1], axis=1), self.policy.condition: np.expand_dims(datas.conditions[:, -1], axis=1)})
                SUMMARY_WRITER.add_summary(summary_str, train_times)
            train_times += 1

def get_data2(env, actor, index=0, train=True, max_choose=False):
    state, condition = env.reset(index=index, train=train)
    last_action = -1
    datas = {'states':[],
             'actions':[],
             'last_actions':[],
             'states_':[],
             'rewards':[],
             'conditions': [],
             'values':[],
             'reward_total':0}

    lstm_state_c, lstm_state_h = actor.get_initial_features(1)

    for _ in range(env.episode_length):
        action, value, lstm_c, lstm_h = actor.choose_action([[state]], [[condition]], [[last_action]], lstm_state_c, lstm_state_h)
        state_, reward, done, info = env.step([action])

        datas['states'].append(state)
        datas['last_actions'].append(last_action)
        datas['actions'].append(action)
        datas['conditions'].append(condition)
        datas['states_'].append(state_)
        datas['rewards'].append(reward)
        datas['values'].append([value])
        datas['reward_total'] += reward
        state = state_
        last_action = action
        lstm_state_c = lstm_c
        lstm_state_h = lstm_h
    datas['values'].append([0])
    return datas


def get_train_data(env, policy, index=0):
    batch_states = []
    batch_actions = []
    batch_last_actions = []
    batch_states_ = []
    batch_rewards = []
    batch_conditions = []
    batch_values = []
    reward_total = 0
    for _ in range(BATCH_SIZE):
        for i in range(env.batch_size):
            #datas = get_data2(env, policy, i, max_choose=False)
            state, condition = env.reset(index=i, train=True)
            datas = policy.predict_detail(state, condition, False)

            batch_states.append(datas['states'])
            batch_actions.append(datas['actions'])
            batch_last_actions.append(datas['last_actions'])
            batch_states_.append(datas['states_'])
            batch_rewards.append(datas['rewards'])
            batch_conditions.append(datas['conditions'])
            batch_values.append(datas['values'])
            reward_total += datas['reward_total']
    batch_states = np.array(batch_states)
    batch_actions = np.array(batch_actions)
    batch_last_actions = np.array(batch_last_actions)
    batch_states_ = np.array(batch_states_)
    batch_rewards = np.array(batch_rewards)
    batch_conditions = np.array(batch_conditions)
    batch_values = np.array(batch_values)
    return Batch(batch_states, batch_actions, batch_last_actions, batch_states_, batch_rewards, batch_conditions, batch_values, reward_total)

def get_data(policy, env, batch_size, train=True, chooce_max=False, fixed=False):
    batch_states = []
    batch_actions = []
    batch_last_actions = []
    batch_states_ = []
    batch_rewards = []
    batch_conditions = []
    batch_values = []
    reward_total = 0
    for index in range(batch_size):
        get_index = None
        if fixed:
            get_index = index

        state, condition = env.reset(index=get_index, train=train)
        datas = policy.predict_detail(state, condition, chooce_max)

        batch_states.append(datas['states'])
        batch_actions.append(datas['actions'])
        batch_last_actions.append(datas['last_actions'])
        batch_states_.append(datas['states_'])
        batch_rewards.append(datas['rewards'])
        batch_conditions.append(datas['conditions'])
        batch_values.append(datas['values'])
        reward_total += datas['reward_total']

    batch_states = np.array(batch_states)
    batch_actions = np.array(batch_actions)
    batch_last_actions = np.array(batch_last_actions)
    batch_states_ = np.array(batch_states_)
    batch_rewards = np.array(batch_rewards)
    batch_values = np.array(batch_values)
    batch_conditions = np.array(batch_conditions)

    return Batch(batch_states, batch_actions, batch_last_actions, batch_states_, batch_rewards, batch_conditions, batch_values, reward_total)


if __name__ == "__main__":

    SESS = tf.Session()

    with tf.device("/cpu:0"):
        GLOBAL_P = Policy(env=env,state_shape=N_F, n_actions=N_A, scope=GLOBAL_POLICY_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        workers.append(SummerWorker(GLOBAL_P))
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_P))

    COORD = tf.train.Coordinator()
    SUMMARY_OP = tf.summary.merge_all()
    SESS.run(tf.global_variables_initializer())
    SUMMARY_WRITER = tf.summary.FileWriter("log/", graph=SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
