import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear


from configs.q3_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
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
        # print num_actions
        out = state
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
            # print conv3.get_shape()
            flattened_filters = tf.reshape(conv3, [-1, 15*15*64])
            hidden = tf.layers.dense(inputs=flattened_filters,units=512, activation=tf.nn.relu)
            if scope == 'q':
                self.conv = []
                self.conv.append(conv1)
                self.conv.append(conv2)
                self.conv.append(conv3)
                self.conv_out = flattened_filters
                self.hidden = hidden
            # print flattened_filters.get_shape()

            W =  tf.get_variable("weights", (512,num_actions))
            b = tf.get_variable("biases", (num_actions))
            out = tf.matmul(hidden,W)+b
            # print out.get_shape()
        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
