import gym
import os
import gym
import numpy as np
import logging
import time
import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear
import scipy
from configs.predict_features import config
from NatureQN import NatureQN


from gym import wrappers
from collections import deque

from utils.general import get_logger, Progbar, export_plot
from utils.replay_buffer import ReplayBuffer
from utils.wrappers import PreproWrapper, MaxAndSkipEnv
from pong_feature_extractor import PongExtractor

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
class FeatureReproducer(object):
    def __init__(self,env,config,DQN,extractor):
        self.feature_extractor = extractor
        self.env = env
        self.config = config
        self.DQN = DQN
        self.DQN.initialize()
        self.build()

    def add_placeholders_op(self):
        self.inputs = tf.placeholder(tf.float32,shape=(None,self.DQN.conv_out.get_shape()[1]))
        self.target = tf.placeholder(tf.float32,shape=(None,self.feature_extractor.feature_length))
        self.lr = tf.placeholder(tf.float32,shape=None)

    def initialize(self):
        # varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,'predictor')
        # for v in varlist:
        #     print v.name
        varlist = tf.all_variables()
        init = tf.variables_initializer(varlist)
        self.sess = tf.Session()
        self.sess.run(init)

    def add_prediction_op(self):
        numFeats = self.feature_extractor.feature_length
        with tf.variable_scope('predictor'):
            h1 = tf.layers.dense(inputs=self.inputs,units=128, activation=tf.nn.relu)
            h2 = tf.layers.dense(inputs=h1,units=128, activation=tf.nn.relu)
            W =  tf.get_variable("weights", (128,numFeats))
            b = tf.get_variable("biases", (numFeats))
            out = tf.matmul(h2,W)+b
        return out

    def add_loss_op(self,p):
        numFeats = self.feature_extractor.feature_length
        with tf.variable_scope('predictor'):
            loss = tf.square(self.target - p)
            self.loss = tf.reduce_mean(loss)

    def add_optimizer_op(self):
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='predictor')
        opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = opt.minimize(self.loss,var_list=train_vars)

    def build(self):
        self.add_placeholders_op()
        self.p = self.add_prediction_op()
        self.add_loss_op(self.p)
        self.add_optimizer_op()
        # print p
        # print self.loss
        # print self.train_op

    def train_step(self,inputs,targets,lr):
        fd = {
            # inputs
            self.inputs: inputs,
            self.target: targets,
            self.lr: lr,
        }

        loss_eval,pred ,_ = self.sess.run([self.loss, self.p,self.train_op], feed_dict=fd)
        return loss_eval,pred

    def train(self,lr_schedule):
        # arguments defaults
        env = self.env

        # replay memory to play
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = []
        t = 0
        losses = deque(maxlen=100)
        err_storage = deque(maxlen=200)
        np.set_printoptions(precision=3)
        for i in range(1000000):
            total_reward = 0
            state = self.env.reset()
            while True:
                t+=1
                lr_schedule.update(t)

                targets = extractor.extract(state)
                targets.shape = (1,8)

                if self.config.render_test: env.render()
                # store last state in buffer
                idx     = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                inputs = self.DQN.sess.run(self.DQN.conv_out, feed_dict={self.DQN.s: [q_input]})
                # print inputs.shape
                # print inputs.shape

                loss_eval,pred = self.train_step(inputs,targets, lr_schedule.epsilon)
                losses.append(loss_eval)

                errs = np.absolute(targets-pred)
                err_storage.append(errs)
                if t % 100 == 0:
                    sys.stdout.write('\r')
                    sys.stdout.flush()
                    mean_err = np.mean(np.vstack(list(err_storage)),axis=0)
                    print t,"Mean Loss: ", np.mean(np.array(list(losses))),mean_err

                # if t % 500 == 0:
                #     print "Real,Predicted"
                #     print targets
                #     print pred
                #     print np.absolute(targets-pred)

                # Advance State
                action = self.DQN.get_action(q_input)
                # print action

                # perform action in env
                new_state, reward, done, info = self.env.step(action)

                # store in replay memory
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # count reward
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
            self.logger.info(msg)

        return avg_reward

    def run(self,exp_schedule,lr_schedule):

        self.initialize()

        self.train(lr_schedule)

if __name__ == '__main__':
    # make env
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                        overwrite_render=config.overwrite_render)

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    DQN_Network = NatureQN(env, config)
    extractor = PongExtractor()
    reproducer = FeatureReproducer(env,config,DQN_Network,extractor)
    reproducer.run(exp_schedule, lr_schedule)
