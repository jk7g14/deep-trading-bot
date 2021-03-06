"""
This part of code is the Deep Q Network (DQN) brain.

view the tensorboard picture about this DQN structure on: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: r1.2
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=320,
            memory_size=100000,
            batch_size=64,
            e_greedy_increment=0.001,
            output_graph=True,
            split_size=9,
            window_size=20,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.split_size = split_size
        self.window_size = window_size
        self.channels = 5
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.5 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):

            inputs = tf.reshape(self.s,
                        shape=[tf.shape(self.s)[0],
                               self.split_size,
                               self.window_size,
                               self.channels])

            c1 = tf.layers.conv2d(inputs=inputs,
                                  filters=10,
                                  kernel_size=[1,5],
                                  strides=(1,1),
                                  padding='same',
                                  activation=tf.nn.relu,
                                  name='c1')

            c2 = tf.layers.conv2d(inputs=c1,
                                  filters=20,
                                  kernel_size=[1,3],
                                  strides=(1,1),
                                  padding='same',
                                  activation=tf.nn.relu,
                                  name='c2')
            c2 = tf.reshape(c2,
                        shape=[tf.shape(self.s)[0],
                               self.split_size,
                               self.window_size*20])
            gru_cells = []
            for i in range(2):
                cell = tf.nn.rnn_cell.GRUCell(
                    num_units=128,
                    name="g%d"%i)
                gru_cells.append(cell)
            multicell = tf.nn.rnn_cell.MultiRNNCell(gru_cells)
            output, final_state = tf.nn.dynamic_rnn(
                cell=multicell,
                inputs=c2,
                dtype=tf.float32
            )
            output = tf.unstack(output, axis=1)[-1]

            e1 = tf.layers.dense(output, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')
            #e3 = tf.layers.dense(e2, 128, tf.nn.sigmoid, kernel_initializer=w_initializer,
            #                     bias_initializer=b_initializer, name='e3')
            self.q_eval = tf.layers.dense(e2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            inputs = tf.reshape(self.s_,
                        shape=[tf.shape(self.s_)[0],
                               self.split_size,
                               self.window_size,
                               self.channels])

            c1 = tf.layers.conv2d(inputs=inputs,
                                  filters=10,
                                  kernel_size=[1,5],
                                  strides=(1,1),
                                  padding='same',
                                  activation=tf.nn.relu,
                                  name='tc1')

            c2 = tf.layers.conv2d(inputs=c1,
                                  filters=20,
                                  kernel_size=[1,3],
                                  strides=(1,1),
                                  padding='same',
                                  activation=tf.nn.relu,
                                  name='tc2')
            c2 = tf.reshape(c2,
                        shape=[tf.shape(self.s_)[0],
                               self.split_size,
                               self.window_size*20])
            gru_cells = []
            for i in range(2):
                cell = tf.nn.rnn_cell.GRUCell(
                    num_units=128,
                    name="tg%d"%i)
                gru_cells.append(cell)
            multicell = tf.nn.rnn_cell.MultiRNNCell(gru_cells)
            output, final_state = tf.nn.dynamic_rnn(
                cell=multicell,
                inputs=c2,
                dtype=tf.float32
            )
            output = tf.unstack(output, axis=1)[-1]

            t1 = tf.layers.dense(output, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')

            t2 = tf.layers.dense(t1, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')
            #t3 = tf.layers.dense(t2, 128, tf.nn.sigmoid, kernel_initializer=w_initializer,
            #                     bias_initializer=b_initializer, name='t3')
            self.q_next = tf.layers.dense(t2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t3')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess,'save/test1.ckpt')
        print('your weight is saved')

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            #print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

if __name__ == '__main__':
    DQN = DeepQNetwork(3,4, output_graph=True)