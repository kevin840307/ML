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
from policy import Policy
import imageio
from collections import namedtuple

tc = tf.nn.rnn_cell
Batch = namedtuple("Batch", ["states", "actions", "last_actions", "states_", "rewards", "conditions", "values", "reward_total"])
LR = 2e-5
OPT = tf.train.AdamOptimizer(LR)
N_WORKERS = multiprocessing.cpu_count()
GLOBAL_POLICY_NET_SCOPE = 'Global_Policy'
env = Toy()
N_F = env.observation_shape
N_A = 16
MAX_EPISODE = 3000000
MAX_EP_STEPS = 20

def get_data(policy, batch_size, train=True, chooce_max=False, fixed=False):
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

        state, condition = policy.env.reset(index=get_index, train=train)

        datas = policy.predict_detail(state, condition, chooce_max, session=SESS)

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

def get_pull_global(global_policy, local_policy):
    return [l_p.assign(g_p) for l_p, g_p in zip(local_policy.var_list, global_policy.var_list)]

def get_update_global(global_policy, local_policy):
    grads = tf.gradients(local_policy.loss , local_policy.var_list)
    return OPT.apply_gradients(zip(grads, global_policy.var_list))

class Worker(object):
    def __init__(self, name, globalP):
        self.env = Toy()
        self.name = name
        self.policy = Policy(name + '/Policy',env=self.env, state_shape=self.env.observation_shape, n_actions=16)
        self.policy.build()

        self.pull_global_op = get_pull_global(globalP, self.policy)
        self.update_global_op = get_update_global(globalP, self.policy)

    def work(self):
        train_times = 0
        while not COORD.should_stop() and train_times < MAX_EPISODE:
            datas = get_data(self.policy, 3, fixed=True, chooce_max=False)
            loss = self.train(datas.states, datas.rewards, datas.conditions, datas.actions, datas.last_actions, datas.values)
            self.pull_global()
            #if train_times % MAX_EP_STEPS == 0:
            #    datas = get_data(self.globalP, self.env, 3)
            #    print(datas.reward_total)
            #    print(self.name + ' time:', train_times,
            #            ' reward_total:', datas.reward_total,
            #            ' mean_reward:', datas.reward_total / BATCH_SIZE)
            train_times += 1

    def train(self, state, reward, condition, action, last_action, value):
        policy = self.policy
        lstm_state_c, lstm_state_h = policy.get_initial_features(len(state))
        value_ = value[:,1:]
        session = SESS
        feed_dict = {policy.last_action:last_action, policy.state: state, policy.action: action, policy.value_: value_, policy.reward: reward, policy.condition: condition, policy.init_state[0]:lstm_state_c, policy.init_state[1]:lstm_state_h}
        loss, _ = session.run([self.policy.loss, self.update_global_op], feed_dict=feed_dict)
        return loss

    def pull_global(self):
        session = SESS
        session.run(self.pull_global_op)

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
                datas = get_data(self.policy, 3, fixed=True, chooce_max=False)
                print('time:', train_times,
                        ' train_reward_total:', datas.reward_total)

                summary_str = SESS.run(SUMMARY_OP, feed_dict={self.policy.state: np.expand_dims(datas.states_[:, -1], axis=1), self.policy.condition: np.expand_dims(datas.conditions[:, -1], axis=1)})
                SUMMARY_WRITER.add_summary(summary_str, train_times)
            train_times += 1


if __name__ == "__main__":

    with tf.device("/cpu:0"):
        GLOBAL_P = Policy(GLOBAL_POLICY_NET_SCOPE, env=env,state_shape=N_F, n_actions=N_A)  # we only need its params
        workers = []
        # Create worker
        workers.append(SummerWorker(GLOBAL_P))
        GLOBAL_P.build(show_img=True)
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_P))

    COORD = tf.train.Coordinator()
    SUMMARY_OP = tf.summary.merge_all()
    GLOBAL_P.complete()
    SESS = tf.get_default_session()

    SUMMARY_WRITER = tf.summary.FileWriter("log/", graph=SESS.graph)
    worker_threads = []


    for worker in workers:
        job = lambda: worker.work()
        thread = threading.Thread(target=job)
        thread.start()
        worker_threads.append(thread)

    COORD.join(worker_threads)