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
from configs.features_dql import config
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
class FeatureExtractedQN(NatureQN):
        """
        Implementing DeepMind's Nature paper. Here are the relevant urls.
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        """

        def add_placeholders_op(self):
            """
            Adds placeholders to the graph

            These placeholders are used as inputs by the rest of the model building and will be fed
            data during training.  Note that when "None" is in a placeholder's shape, it's flexible
            (so we can use different batch sizes without rebuilding the model
            """
            # this information might be useful
            # here, typically, a state shape is (80, 80, 1)
            # state_shape = list(self.env.observation_space.shape)

            ##############################################################
            """
            TODO: add placeholders:
                  Remember that we stack 4 consecutive frames together, ending up with an input of shape
                  (80, 80, 4).
                   - self.s: batch of states, type = uint8
                             shape = (batch_size, img height, img width, nchannels x config.state_history)
                   - self.a: batch of actions, type = int32
                             shape = (batch_size)
                   - self.r: batch of rewards, type = float32
                             shape = (batch_size)
                   - self.sp: batch of next states, type = uint8
                             shape = (batch_size, img height, img width, nchannels x config.state_history)
                   - self.done_mask: batch of done, type = bool
                             shape = (batch_size)
                             note that this placeholder contains bool = True only if we are done in
                             the relevant transition
                   - self.lr: learning rate, type = float32

            (Don't change the variable names!)

            HINT: variables from config are accessible with self.config.variable_name
                  Also, you may want to use a dynamic dimension for the batch dimension.
                  Check the use of None for tensorflow placeholders.

                  you can also use the state_shape computed above.
            """
            ##############################################################
            ################YOUR CODE HERE (6-15 lines) ##################

            # print "state shape",state_shape
            self.s = tf.placeholder(tf.uint8,  shape= (None,\
                                                        8,\
                                                        1,\
                                                        1*self.config.state_history)\
                                                        )
            self.a = tf.placeholder(tf.int32,  shape= (None))
            self.r = tf.placeholder(tf.float32,  shape= (None))
            self.sp = tf.placeholder(tf.uint8,  shape= (None,\
                                                        8,\
                                                        1,\
                                                        1*self.config.state_history)\
                                                        )
            self.done_mask = tf.placeholder(tf.bool,  shape= (None))
            self.lr = tf.placeholder(tf.float32,shape=None)
            ##############################################################
            ######################## END YOUR CODE #######################


        def get_q_values_op(self, state, scope, reuse=False):
            """
            Returns Q values for all actions

            Args:
                state: (tf tensor)
                    shape = (batch_size, img height, img width, nchannels)
                scope: (string) scope name, that specifies if target network or not
                reuse: (bool) reuse of variables in the scope

            Returns:
                out: (tf tensor) of shape = (batch_size, num_actions)
            """
            # this information might be useful
            num_actions = self.env.action_space.n
            # print num_actions
            # out = state
            ##############################################################
            """
            TODO: implement the computation of Q values like in the paper
                    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

                  you may find the section "model architecture" of the appendix of the
                  nature paper particulary useful.

                  store your result in out of shape = (batch_size, num_actions)

            HINT: you may find tensorflow.contrib.layers useful (imported)
                  make sure to understand the use of the scope param

                  you can use any other methods from tensorflow
                  you are not allowed to import extra packages (like keras,
                  lasagne, cafe, etc.)

            """
            ##############################################################
            ################ YOUR CODE HERE - 10-15 lines ################
            with tf.variable_scope(scope) as sc:

                inflow =tf.reshape(state, [-1, 8])
                h1 = tf.layers.dense(inputs=inflow,units=64, activation=tf.nn.relu)
                h2 = tf.layers.dense(inputs=h1,units=32, activation=tf.nn.relu)
                # h3 = tf.layers.dense(inputs=h1,units=64, activation=tf.nn.relu)

                W =  tf.get_variable("weights", (32,num_actions))
                b = tf.get_variable("biases", (num_actions))
                out = tf.matmul(h2,W)+b
                # print out.get_shape()
            ##############################################################
            ######################## END YOUR CODE #######################
            return out

        def train(self, exp_schedule, lr_schedule):
            """
            Performs training of Q

            Args:
                exp_schedule: Exploration instance s.t.
                    exp_schedule.get_action(best_action) returns an action
                lr_schedule: Schedule for learning rate
            """

            # initialize replay buffer and variables
            replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
            rewards = deque(maxlen=self.config.num_episodes_test)
            max_q_values = deque(maxlen=1000)
            q_values = deque(maxlen=1000)
            self.init_averages()

            t = last_eval = last_record = 0 # time control of nb of steps
            scores_eval = [] # list of scores computed at iteration time
            scores_eval += []

            extractor = PongExtractor()

            prog = Progbar(target=self.config.nsteps_train)

            # interact with environment
            while t < self.config.nsteps_train:
                total_reward = 0
                frame = self.env.reset()
                state = extractor.extract(frame)
                state = state[:,np.newaxis,np.newaxis]
                # print state.shape
                while True:
                    t += 1
                    last_eval += 1
                    last_record += 1
                    if self.config.render_train: self.env.render()
                    # replay memory stuff
                    idx      = replay_buffer.store_frame(state)
                    q_input = replay_buffer.encode_recent_observation()

                    # chose action according to current Q and exploration
                    best_action, q_values = self.get_best_action(q_input)
                    action                = exp_schedule.get_action(best_action)

                    # store q values
                    max_q_values.append(max(q_values))
                    q_values += list(q_values)

                    # perform action in env
                    new_frame, reward, done, info = self.env.step(action)
                    new_state = extractor.extract(new_frame)
                    new_state = new_state[:,np.newaxis,np.newaxis]
                    # print new_state.shape
                    # print "state shape:",state.shape()

                    # store the transition
                    replay_buffer.store_effect(idx, action, reward, done)
                    state = new_state

                    # perform a training step
                    loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.epsilon)

                    # logging stuff
                    if ((t > self.config.learning_start) and (t % self.config.log_freq == 0) and
                       (t % self.config.learning_freq == 0)):
                        self.update_averages(rewards, max_q_values, q_values, scores_eval)
                        exp_schedule.update(t)
                        lr_schedule.update(t)
                        if len(rewards) > 0:
                            prog.update(t + 1, exact=[("Loss", loss_eval), ("Avg R", self.avg_reward),
                                            ("Max R", np.max(rewards)), ("eps", exp_schedule.epsilon),
                                            ("Grads", grad_eval), ("Max Q", self.max_q),
                                            ("lr", lr_schedule.epsilon)])

                    elif (t < self.config.learning_start) and (t % self.config.log_freq == 0):
                        sys.stdout.write("\rPopulating the memory {}/{}...".format(t,
                                                            self.config.learning_start))
                        sys.stdout.flush()

                    # count reward
                    total_reward += reward
                    if done or t >= self.config.nsteps_train:
                        break

                # updates to perform at the end of an episode
                rewards.append(total_reward)

                if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                    # evaluate our policy
                    last_eval = 0
                    print("")
                    scores_eval += [self.evaluate()]

                if (t > self.config.learning_start) and self.config.record and (last_record > self.config.record_freq):
                    self.logger.info("Recording...")
                    last_record =0
                    self.record()

            # last words
            self.logger.info("- Training done.")
            self.save()
            scores_eval += [self.evaluate()]
            export_plot(scores_eval, "Scores", self.config.plot_output)

        def evaluate(self, env=None, num_episodes=None):
            """
            Evaluation with same procedure as the training
            """
            extractor = PongExtractor()
            # log our activity only if default call
            if num_episodes is None:
                self.logger.info("Evaluating...")

            # arguments defaults
            if num_episodes is None:
                num_episodes = self.config.num_episodes_test

            if env is None:
                env = self.env

            # replay memory to play
            replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
            rewards = []

            for i in range(num_episodes):
                total_reward = 0
                frame = self.env.reset()
                state = extractor.extract(frame)
                state = state[:,np.newaxis,np.newaxis]
                while True:
                    if self.config.render_test: env.render()

                    # store last state in buffer
                    idx     = replay_buffer.store_frame(state)
                    q_input = replay_buffer.encode_recent_observation()

                    action = self.get_action(q_input)

                    # perform action in env
                    new_frame, reward, done, info = self.env.step(action)
                    new_state = extractor.extract(new_frame)
                    new_state = new_state[:,np.newaxis,np.newaxis]

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
    model = FeatureExtractedQN(env, config)
    model.run(exp_schedule, lr_schedule)
