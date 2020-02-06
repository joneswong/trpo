import gym
import numpy as np
from gym.spaces import Discrete, Box, Tuple


class Sim0(gym.Env):

    def __init__(self, num_arms=2):
        super(Sim0, self).__init__()
        self._num_arms = num_arms
        self.rw = np.eye(num_arms, dtype=np.float32)
        self.cw = np.eye(num_arms, dtype=np.float32)
        self.observation_space = Box(0.0, 1.0, [num_arms])
        self.action_space = Discrete(num_arms)

    def reset(self):
        self.context = np.random.rand(self._num_arms)
        return self.context

    def step(self, act):
        reward = np.sum(self.rw[act] * self.context)
        constraint = -1.0 * np.sum(self.cw[act] * (1.0-self.context))
        next_ob = np.random.rand(self._num_arms)
        done = True
        return next_ob, reward, done, {"c0": constraint}


if __name__=="__main__":
    env = Sim0(6)
    ob = env.reset()
    for _ in range(100):
        next_ob, rwd, done, info = env.step(env.action_space.sample())
        print("{} v.s. {}".format(rwd, info["c0"]))
