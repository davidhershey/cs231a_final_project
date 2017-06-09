import gym
import os
import gym
import numpy as np
import logging
import time
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear
import scipy
from configs.predict_features import config

from configs.load_breakout import config as config2
from NatureQN import NatureQN


from gym import wrappers
from collections import deque

from utils.general import get_logger, Progbar, export_plot
from utils.replay_buffer import ReplayBuffer
from utils.wrappers import PreproWrapper, MaxAndSkipEnv
from pong_feature_extractor import PongExtractor

from q6_bonus_question import MyDQN


"""
Use linear approximation for the Atari game. Please report the final result.
Feel free to change the configurations (in the configs/ folder).
If so, please report your hyperparameters.

You'll find the results, log and video recordings of your agent every 250k under
the corresponding file in the results folder. A good way to monitor the progress
of the training is to use Tensorboard. The starter code writes summaries of different
variables.

To launch tensorboard, open a Terminal window and run
tensorboard --logdir=results/
Then, connect remotely to
address-ip-of-the-server:6006
6006 is the default port used by tensorboard.
"""
def getMap(DQN,layer,q_input,prefix=""):

    activation = np.squeeze(DQN.sess.run(layer,{DQN.s: [q_input]}))
    print activation.shape

    for i in range(activation.shape[2]):
        act_map = activation[:,:,i]
        plt.imshow(act_map, cmap='hot', interpolation='nearest')
        plt.savefig('activations/breakout/conv3/{}filter{}.png'.format(prefix,i))
        # print act_map.shape
    # for image in images:
    #     image =
    #     print image.shape
    #     image = image[np.newaxis,:,:]
    #     print image.shape
    # replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)

    pass
import pickle
if __name__ == '__main__':
    # make env
    frame = 149
    pong = False
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                        overwrite_render=config.overwrite_render)
    # train model
    q_input = np.load('embeddings/breakoutA/breakout_input{}.npy'.format(frame))
    if pong:
        DQN = NatureQN(env, config)
    else:
        DQN = MyDQN(env,config2)
    DQN.initialize()
    # layer = 1
    activation = getMap(DQN,DQN.conv[2], q_input,"frame_{}_".format(frame))
