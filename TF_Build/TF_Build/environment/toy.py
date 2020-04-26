import gym
import numpy as np
import sys
import cv2
from Environment import Environment

import imageio


class Toy(gym.Env, Environment):
    action_sizes = {'position': [16]}

    metadata = {
        'render.modes': ['rgb_array'],
    }

    colors = [(0., 0., 0.), # black
            (102., 217., 232.), # cyan 3
            (173., 181., 189.), # gray 5
            (255., 224., 102.), # yellow 3
            (229., 153., 247.), # grape 3
            (99., 230., 190.), # teal 3
            (255., 192., 120.), # orange 3
            (255., 168., 168.), # red 3
        ]


    def __init__(self):
        """ init env """
        gym.Env.__init__(self)
        Environment.__init__(self)
        self.width, self.height, self.channel = 16, 16, 1
        self.episode_length = 3
        self.batch_size = 3
        self.observation_shape = [self.height, self.width, self.channel]
        self.observation_space = self.height * self.width * self.channel
        self._step = 0
        #self.train_patterns = [(1, 3, 5, 7, 9, 11, 13, 15)]
        #self.train_patterns = [(0, 2, 4, 6, 8, 10), (1, 3, 5, 7, 9, 11), (2,
        #4, 6, 8, 10, 12)]
        self.train_patterns = [[0, 2, 4], [1, 3, 5], [2, 4, 6]]
        #self.train_patterns = [[1], [11], [4]]
        #self.train_patterns = [[1, 3, 5], [11, 13, 15], [5, 7, 9]]
        #self.train_patterns = [[1, 3, 5], [11, 13, 15], [10, 12, 14]]
        #self.train_patterns = [[1], [7], [14]]
        #self.train_patterns = [[1, 3, 5], [11, 13, 15], [10, 12, 14], [3, 5,
        #7], [4, 6, 8], [7, 9, 11]]
        #self.train_patterns = np.random.randint(low=0, high=15, size=(100, 3))
        #self.test_patterns = np.random.randint(low=0, high=15, size=(100, 3))
        #self.test_patterns = [[2], [5]]
        self.test_patterns = [[3, 5, 7], [4, 6, 8]]
        #self.test_patterns = [[2], [7], [9]]
        #self.test_patterns = [[3, 5, 7, 9, 11, 13], [4, 6, 8, 10, 12, 14]]
        self.train, self.test = self.__get_data()
        self.reset()

    def __get_data(self):
        train = [self.__create_target_data(pattern) for pattern in self.train_patterns]
        test = [self.__create_target_data(pattern) for pattern in self.test_patterns]
        return train, test

    def __create_target_data(self, pattern):
        self.image = np.ones((self.width, self.height, 3)) * 255
        self.image = self.image.astype(np.uint8)
        for index in pattern:
            a = [index,  1.0,  0,  1]
            self.step(a, init=True)
        return self.render('gray_array')

    def __len__(self):
        return len(self.train)

    def step(self, action, init=False):
        x = action[0]

        p1, p2 = self.convert_x(x)

        #self.image[p1, p2, :] = 0
        #self.image[p2, p1, :] = 0
        self.image[p1, :, :] = 0
        self.image[:, p2, :] = 0
        self.image[p2, :, :] = 0
        self.image[:, p1, :] = 0

        self.state = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.state = self.norm(self.state)
        self.state = self.state .reshape(self.width, self.height, 1)


        self._step += 1
        terminal = (self._step == self.episode_length)
        reward = 0
        if terminal and not init:
            reward = 1
        #if not init:
            #reward += -l2(self.state, self.random_target) /
            #np.prod(self.observation_shape) * 100
            #reward += - l2(self.denorm(self.state),
            #self.denorm(self.random_target)) / self.observation_space
        #reward += - l1(self.state, self.random_target) /
        #self.observation_space
            #reward += - l2(self.state, self.random_target) /
            #self.observation_space
            reward +=  - l2(self.state, self.random_target) / ((self.height + self.width) / 2)

            #reward += - l2(self.state, self.random_target) /
            #self.episode_length / 2
        #imsave1('Image/123.png', self.state)
        #imsave1('Image/125.png', self.random_target)
        # return observation, reward, done, and info
        return self.state, reward, terminal, {}

    def reset(self, index=None, train=True):
        self.image = np.ones((self.width, self.height, 3)) * 255.0
        self.image = self.image.astype(np.uint8)

        # return observation, reward, done, and info
        self._step = 0
        self.random_target = self.get_example(num=1, train=train)
        self.random_target = np.reshape(self.random_target, (self.width, self.height, 1))
        if index != None:
            if train:
                self.random_target = self.train[index]
            else:
                self.random_target = self.test[index]

        
        self.state = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.state = self.norm(self.state)
        self.state = self.state .reshape(self.width, self.height, 1)

        return self.state, self.random_target

    def get_example(self, num=1, train=True):
        datas = []
        if train:
            for index in range(num):
                i = np.random.randint(len(self.train))
                datas.append(self.train[i])
        else:
            for index in range(num):
                i = np.random.randint(len(self.test))
                datas.append(self.test[i])
        return np.array(datas)

    def convert_x(self, x):
        """ convert position id -> a point (p1, p2) """
        assert x < self.width * self.height
        p1 = x % self.width
        p2 = x // self.height
        return int(p1), int(p2)

    def render(self, mode='rgb_array'):
        """ render the current drawn picture image for human """
        if mode == 'rgb_array':
            return self.image
        elif mode == 'gray_array':
            return self.state
        else:
            raise NotImplementedError

    def close(self):
        pass

    def seed(self, seed=None):
        pass

def l2(mat1, mat2):
    return np.sqrt(np.sum((mat1 - mat2) ** 2))

def l1(mat1, mat2):
    return np.sum(np.abs(mat1 - mat2))

def imsave1(path, img):
    img = np.clip(img * 127.5 + 127.5, 0, 255).astype(np.uint8)
    img = np.tile(img, [1, 1, 3])
    imageio.imwrite(path, img)
