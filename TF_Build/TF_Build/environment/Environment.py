import numpy as np


class Environment(object):

    def __init__(self):
        # terminal
        self.episode_length = 2

        # screen
        self.height, self.width, self.channel = 16, 16, 1
        self.observation_space = [self.height, self.width, self.channel]

        for name, value in self.action_sizes.items():
            if value is None:
                self.action_sizes[name] = self.location_shape

        self.acs = list(self.action_sizes.keys())
        self.ac_idx = {
                ac:idx for idx, ac in enumerate(self.acs)
        }


    def random_action(self):
        action = []
        for ac in self.acs:
            size = self.action_sizes[ac]
            sample = np.random.randint(np.prod(size))
            action.append(sample)
        return action

    @property
    def initial_action(self):
        return [0] * len(self.acs)

    def norm(self, img):
        return (np.array(img) - 127.5) / 127.5

    def denorm(self, img):
        return img * 127.5 + 127.5