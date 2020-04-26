import sys, os
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

from toy import Toy
from actor import Actor
from critic import Critic
from module import Model
import tensorflow as tf
import tensorflow_tools as tf_tools
import numpy as np
from collections import namedtuple
Batch = namedtuple("Batch", ["states", "actions", "last_actions", "states_", "rewards", "conditions", "reward_total"])

def get_data(actor, env, batch_size, train=True, chooce_max=False, fixed=False):
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
        datas = actor.predict_detail(state, condition, chooce_max)

        batch_states.append(datas['states'])
        batch_actions.append(datas['actions'])
        batch_last_actions.append(datas['last_actions'])
        batch_states_.append(datas['states_'])
        batch_rewards.append(datas['rewards'])
        batch_conditions.append(datas['conditions'])
        reward_total += datas['reward_total']

    batch_states = np.array(batch_states)
    batch_actions = np.array(batch_actions)
    batch_last_actions = np.array(batch_last_actions)
    batch_states_ = np.array(batch_states_)
    batch_rewards = np.array(batch_rewards)
    batch_conditions = np.array(batch_conditions)

    return Batch(batch_states, batch_actions, batch_last_actions, batch_states_, batch_rewards, batch_conditions, reward_total)

class A2C(Model):
    def __init__(self, scope_name, env, states_shape, n_actions, channel_min, channel_rate=2):
        super(A2C, self).__init__(scope_name)
        self.scope_name = scope_name
        self.channel_min = channel_min
        self.channel_rate = channel_rate
        self.states_shape = states_shape
        self.n_actions = n_actions
        self.env = env

    def buind(self, action_lr=1e-5, critic_lr=1e-5):
        self.actor = Actor('Actor',env=self.env,states_shape=self.states_shape, n_actions=self.n_actions)
        self.actor.build(lr=action_lr)

        self.critic = Critic('Critic', env=self.env,states_shape=self.states_shape, n_features=self.n_actions)  
        self.critic.build(lr=critic_lr)

    def predict(self, z):
        session = tf.get_default_session()
        feed_dict = {self.input_z: z}
        output = session.run(self.fake_img, feed_dict=feed_dict)
        return output

    def train(self, datas):
        session = tf.get_default_session()
        td_error = self.critic.train(datas.states, datas.rewards, datas.conditions)  # gradient = grad[r + gamma * V(s_) - V(s)]
        self.actor.train(datas.states, datas.actions, td_error, datas.conditions)     # true_gradient = grad[logPi(s,a) * td_error]
    


if __name__ == '__main__':
    env = Toy()
    LR_A = 1e-5
    LR_C = 1e-5
    N_F = env.observation_shape
    N_A = 16

    a2c = A2C("A2C", env, N_F, N_A, 64)
    a2c.buind(LR_A, LR_C)
    summary_op = tf.summary.merge_all()

    a2c.complete()
    session = tf.get_default_session()
    summary_writer = tf.summary.FileWriter("log/", graph=session.graph)


    for step in range(1000000):
        datas = get_data(a2c.actor, a2c.env, 3, chooce_max=True)
        a2c.train(datas)
        if step % 20 == 0:
            show_states = np.expand_dims(datas.states_[:, -1], axis=1)
            show_conditions = np.expand_dims(datas.conditions[:, -1], axis=1)

            print('time:', step, ' reward:', datas.reward_total)
            summary_str = session.run(summary_op, feed_dict={a2c.critic.state: show_states, a2c.critic.condition: show_conditions})
            summary_writer.add_summary(summary_str, step)
