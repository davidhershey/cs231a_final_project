import gym
import os
import gym
import numpy as np
import logging
import time
import sys
import pickle

from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear
import scipy
from configs.extract_frames import config
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
class FrameExtractor(NatureQN):

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
        last_frames = deque(maxlen=4)
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
            state = self.env.reset()
            last_frame = state
            last_frames.append(state)
            while True:
                t += 1
                last_eval += 1
                last_record += 1
                if self.config.render_train: self.env.render()

                feats = extractor.extract(np.squeeze(state))
                # replay memory stuff
                idx      = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                # chose action according to current Q and exploration
                best_action, q_values = self.get_best_action(q_input)
                embedding = self.sess.run(self.hidden, feed_dict={self.s: [q_input]})[0]
                action                = exp_schedule.get_action(best_action)

                # store q values
                max_q_values.append(max(q_values))
                q_values += list(q_values)
                if t % 100==0:
                    # print state.shape
                    # frame = np.zeros(np.squeeze(state).shape)
                    # for f in last_frames:
                    #     frame = frame + np.squeeze(f)
                    # frame = frame / len(last_frames)
                    frame = np.squeeze(state)
                    last_frame = np.squeeze(last_frame)
                    pickle.dump(last_frames,open('frames/embedding/atari{}.p'.format(t),'w'))
                    for i in range(4):
                        f = np.squeeze(last_frames[i])
                        scipy.misc.imsave('frames/embedding/atari{}.png'.format(t-3+i),f)

                    # scipy.misc.imsave('frames/atari{}.png'.format(t-1),last_frame)
                    # posfile = open('frames/atari{}.txt'.format(t),'w')
                    # posfile.write('Opp Paddle:\t{}\n'.format(oppY))
                    # posfile.write('Player Paddle:\t{}\n'.format(playerY))
                    # posfile.write('ball x:\t{}\n'.format(ballX))
                    # posfile.write('ball y:\t{}\n'.format(ballY))
                    # posfile.close()
                    np.savetxt('frames/embedding/pong{}.txt'.format(t),feats,fmt='%.2f')


                # perform action in env
                new_state, reward, done, info = self.env.step(action)
                # print "state shape:",state.shape()

                # store the transition
                replay_buffer.store_effect(idx, action, reward, done)
                last_frame = state
                state = new_state
                last_frames.append(state)

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
    model = FrameExtractor(env, config)
    model.run(exp_schedule, lr_schedule)
