import gym
import random
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear

from configs.q6_bonus_question import config


class MyDQN(Linear):
    """
    Going beyond - implement your own Deep Q Network to find the perfect
    balance between depth, complexity, number of parameters, etc.
    You can change the way the q-values are computed, the exploration
    strategy, or the learning rate schedule. You can also create your own
    wrapper of environment and transform your input to something that you
    think we'll help to solve the task. Ideally, your network would run faster
    than DeepMind's and achieve similar performance!

    You can also change the optimizer (by overriding the functions defined
    in TFLinear), or even change the sampling strategy from the replay buffer.

    If you prefer not to build on the current architecture, you're welcome to
    write your own code.

    You may also try more recent approaches, like double Q learning
    (see https://arxiv.org/pdf/1509.06461.pdf) or dueling networks
    (see https://arxiv.org/abs/1511.06581), but this would be for extra
    extra bonus points.
    """
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
        out = state
        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)
        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################

        with tf.variable_scope(scope) as sc:
            if reuse:
                sc.reuse_variables()
            conv1 = tf.layers.conv2d(inputs=state,
                                    filters=32,
                                    kernel_size=[8, 8],
                                    activation=tf.nn.relu,
                                    strides=2)
            conv2 = tf.layers.conv2d(inputs=conv1,
                                    filters=64,
                                    kernel_size=[4,4],
                                    activation=tf.nn.relu,
                                    strides=2)
            conv3 = tf.layers.conv2d(inputs=conv2,
                                    filters=64,
                                    kernel_size=[3,3],
                                    activation=tf.nn.relu,
                                    strides=1)
            # outshape = tf.shape(conv3)
            print conv3.get_shape()
            flattened_filters = tf.reshape(conv3, [-1, 15*15*64])
            # print flattened_filters.get_shape()
            hidden = tf.layers.dense(inputs=flattened_filters,units=512, activation=tf.nn.relu)

            W =  tf.get_variable("weights", (512,num_actions))
            b = tf.get_variable("biases", (num_actions))
            out = tf.matmul(hidden,W)+b
            # print out.get_shape()

        ##############################################################
        ######################## END YOUR CODE #######################
        return out

    def add_loss_op(self, q,qp, target_q,scope):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: The loss for an example is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2

              You need to compute the average of the loss over the minibatch
              and store the resulting scalar into self.loss

        HINT: - config variables are accessible through self.config
              - you can access placeholders like self.a (for actions)
                self.r (rewards) or self.done_mask for instance
              - you may find the following functions useful
                    - tf.cast
                    - tf.reduce_max / reduce_sum
                    - tf.one_hot
                    - ...

        (be sure that you set self.loss)
        """
        ##############################################################
        ##################### YOUR CODE HERE - 4-5 lines #############
        with tf.variable_scope(scope) as sc:
            selected = tf.argmax(qp,axis=1)
            selected_idx = tf.one_hot(selected,num_actions)
            Qs = tf.reduce_sum(tf.multiply(selected_idx,target_q),1)
            future = self.config.gamma * Qs
            flip = tf.logical_not(self.done_mask)
            mask = tf.cast(flip, tf.float32)
            future = tf.multiply(mask,future)
            q_samp = self.r + future
            idx_onehot = tf.one_hot(self.a,num_actions)
            Qa = tf.reduce_sum(tf.multiply(idx_onehot,q),1)
            loss = tf.square(q_samp - Qa)
            # loss = tf.Print(loss,[idx_onehot],first_n=2,summarize=16)
            return tf.reduce_mean(loss)

        ##############################################################
        ######################## END YOUR CODE #######################
    def add_optimizer_op(self,loss, scope):
        """
        Set self.train_op and self.grad_norm
        """

        ##############################################################
        """
        TODO: 1. get Adam Optimizer (remember that we defined self.lr in the placeholders
                section)
              2. compute grads wrt to variables in scope for self.loss
              3. clip the grads by norm with self.config.clip_val if self.config.grad_clip
                is True
              4. apply the gradients and store the train op in self.train_op
               (sess.run(train_op) must update the variables)
              5. compute the global norm of the gradients and store this scalar
                in self.grad_norm

        HINT: you may find the following functinos useful
            - tf.get_collection
            - optimizer.compute_gradients
            - tf.clip_by_norm
            - optimizer.apply_gradients
            - tf.global_norm

             you can access config variable by writing self.config.variable_name

        (be sure that you set self.train_op and self.grad_norm)
        """
        ##############################################################
        #################### YOUR CODE HERE - 8-12 lines #############
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        print loss,scope
        opt = tf.train.AdamOptimizer(self.lr)
        gradients = opt.compute_gradients(loss,train_vars)
        clipped_gradients = []
        for g,v in gradients:
            print g,v
            clipped = tf.clip_by_norm(g,self.config.clip_val)
            clipped_gradients.append(clipped)
        grad_norm = tf.global_norm(clipped_gradients)
        return opt.apply_gradients(zip(clipped_gradients, train_vars)),grad_norm
        ##############################################################
        ######################## END YOUR CODE #######################

    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        a1,a2 = self.sess.run([self.q1,self.q2], feed_dict={self.s: [state]})
        # a2 = self.sess.run(self.q2, feed_dict={self.s: [state]})[0]

        return np.argmax(a1+a2), a1+a2

    def initialize(self):
        """
        Assumes the graph has been constructed
        Creates a tf Session and run initializer of variables
        """
        # create tf session
        self.sess = tf.Session()

        # tensorboard stuff
        self.add_summary()

        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # synchronise q and target_q networks
        # self.sess.run(self.update_target_op)

        # for saving networks weights
        self.saver = tf.train.Saver()

    def build(self):
        """
        Build model by adding all necessary variables
        """
        # add placeholders
        self.add_placeholders_op()

        # compute Q values of state
        s = self.process_state(self.s)
        sp = self.process_state(self.sp)
        self.q1 = self.get_q_values_op(s, scope="q1", reuse=False)
        self.q1p = self.get_q_values_op(sp, scope="q1", reuse=True)

        # compute Q values of next state
        # sp = self.process_state(self.sp)
        self.q2 = self.get_q_values_op(s, scope="q2", reuse=False)
        self.q2p = self.get_q_values_op(sp, scope="q2", reuse=True)

        # add update operator for target network
        # self.add_update_target_op("q", "target_q")

        # add square loss
        self.loss1 = self.add_loss_op(self.q1,self.q1p, self.q2p,"q1")
        self.loss2 = self.add_loss_op(self.q2,self.q2p, self.q1p,"q2")

        # add optmizer for the main networks
        self.train_op1,self.grad_norm1 = self.add_optimizer_op(self.loss1,"q1")
        self.train_op2,self.grad_norm2 = self.add_optimizer_op(self.loss2,"q2")

    def add_summary(self):
        """
        Tensorboard stuff
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self.avg_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="avg_q")
        self.max_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="max_q")
        self.std_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="std_q")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # add placeholders from the graph
        # tf.summary.scalar("loss1", self.loss1)
        # tf.summary.scalar("grads norm1", self.grad_norm1)

        # tf.summary.scalar("loss1", self.loss2)
        # tf.summary.scalar("grads norm1", self.grad_norm2)

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std Reward", self.std_reward_placeholder)

        tf.summary.scalar("Avg Q", self.avg_q_placeholder)
        tf.summary.scalar("Max Q", self.max_q_placeholder)
        tf.summary.scalar("Std Q", self.std_q_placeholder)

        tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)

        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path,
                                                self.sess.graph)

    def update_target_params(self):
        """
        Don't overwrite here
        """
        pass

    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """

        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
            self.config.batch_size)


        fd = {
            # inputs
            self.s: s_batch,
            self.a: a_batch,
            self.r: r_batch,
            self.sp: sp_batch,
            self.done_mask: done_mask_batch,
            self.lr: lr,
            # extra info
            self.avg_reward_placeholder: self.avg_reward,
            self.max_reward_placeholder: self.max_reward,
            self.std_reward_placeholder: self.std_reward,
            self.avg_q_placeholder: self.avg_q,
            self.max_q_placeholder: self.max_q,
            self.std_q_placeholder: self.std_q,
            self.eval_reward_placeholder: self.eval_reward,
        }
        if random.random() < .5:
            print 'loss1'
            loss_eval, grad_norm_eval, summary, _ = self.sess.run([self.loss1, self.grad_norm1,
                                                        self.merged, self.train_op1], feed_dict=fd)
        else:
            print 'loss2'
            loss_eval, grad_norm_eval, summary, _ = self.sess.run([self.loss2, self.grad_norm2,
                                                        self.merged, self.train_op2], feed_dict=fd)


        # tensorboard stuff
        self.file_writer.add_summary(summary, t)

        return loss_eval, grad_norm_eval


"""
Use a different architecture for the Atari game. Please report the final result.
Feel free to change the configuration. If so, please report your hyperparameters.
"""
if __name__ == '__main__':
    # make env
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                        overwrite_render=config.overwrite_render)

    # exploration strategy
    # you may want to modify this schedule
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # you may want to modify this schedule
    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = MyDQN(env, config)
    model.run(exp_schedule, lr_schedule)
