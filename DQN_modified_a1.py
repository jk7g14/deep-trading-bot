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
            e_greedy_increment=None,
            output_graph=False,
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
        self.channels = 9
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.5 if e_greedy_increment is not None else self.epsilon_max

        self.conv_keep_prob = 0.9
        self.gru_keep_prob = 0.5
        self.dense_keep_prob = 0.5

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        #learning rate
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step,
                                           100000, 0.96, staircase=True)

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

        #w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        w_initializer = tf.initializers.glorot_uniform() 

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):

            inputs = tf.reshape(self.s,
                        shape=[tf.shape(self.s)[0],
                               self.split_size,
                               self.window_size,
                               self.channels])

            c1 = tf.layers.conv2d(inputs=inputs,
                                  kernel_initializer=w_initializer,
                                  filters=10,
                                  kernel_size=[1,5],
                                  strides=(1,1),
                                  padding='same',
                                  activation=tf.nn.relu,
                                  name='c1')

            channels = tf.shape(c1)[-1]
            c1 = tf.nn.dropout(c1,
                               keep_prob=self.conv_keep_prob,
                               name='c1d',
                               noise_shape=[tf.shape(self.s)[0], 1, 1, channels])

            c2 = tf.layers.conv2d(inputs=c1,
                                  kernel_initializer=w_initializer,
                                  filters=20,
                                  kernel_size=[1,3],
                                  strides=(1,1),
                                  padding='same',
                                  activation=tf.nn.relu,
                                  name='c2')

            channels = tf.shape(c2)[-1]
            c2 = tf.nn.dropout(c2,
                               keep_prob=self.conv_keep_prob,
                               name='c2d',
                               noise_shape=[tf.shape(self.s)[0], 1, 1, channels])

            c2 = tf.reshape(c2,
                        shape=[tf.shape(self.s)[0],
                               self.split_size,
                               self.window_size*20])
            #GRU
            gru_cells = []
            for i in range(2):
                #GRU CELL
                cell = tf.nn.rnn_cell.GRUCell(
                    kernel_initializer=w_initializer,
                    num_units=128,
                    name="g%d"%i)
                #GRU DROPOUT
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell,
                    output_keep_prob=self.gru_keep_prob,
                    variational_recurrent=True,
                    dtype=tf.float32)

                gru_cells.append(cell)

            multicell = tf.nn.rnn_cell.MultiRNNCell(gru_cells)
            output, final_state = tf.nn.dynamic_rnn(
                cell=multicell,
                inputs=c2,
                dtype=tf.float32
            )
            output = tf.unstack(output, axis=1)[-1]

            e1 = tf.layers.dense(output, 128, tf.nn.relu,
                                 kernel_initializer=w_initializer, name='e1')
            e1 = tf.nn.dropout(e1, keep_prob=self.dense_keep_prob, name='e1d')

            e2 = tf.layers.dense(e1, 64, tf.nn.relu,
                                 kernel_initializer=w_initializer, name='e2')
            e2 = tf.nn.dropout(e2, keep_prob=self.dense_keep_prob, name='e2d')

            e3 = tf.layers.dense(e2, 32, tf.nn.relu,
                                 kernel_initializer=w_initializer, name='e3')

            e3 = tf.nn.dropout(e3, keep_prob=self.dense_keep_prob, name='e3d')

            self.q_eval = tf.layers.dense(e3, self.n_actions,
                                          kernel_initializer=w_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            inputs_ = tf.reshape(self.s_,
                        shape=[tf.shape(self.s_)[0],
                               self.split_size,
                               self.window_size,
                               self.channels])

            tc1 = tf.layers.conv2d(inputs=inputs_,
                                  kernel_initializer=w_initializer,
                                  filters=10,
                                  kernel_size=[1,5],
                                  strides=(1,1),
                                  padding='same',
                                  activation=tf.nn.relu,
                                  name='tc1')

            channels = tf.shape(tc1)[-1]
            tc1 = tf.nn.dropout(tc1,
                               keep_prob=1.0,
                               name='tc1d',
                               noise_shape=[tf.shape(self.s_)[0], 1, 1, channels])

            tc2 = tf.layers.conv2d(inputs=tc1,
                                  kernel_initializer=w_initializer,
                                  filters=20,
                                  kernel_size=[1,3],
                                  strides=(1,1),
                                  padding='same',
                                  activation=tf.nn.relu,
                                  name='tc2')

            channels = tf.shape(tc2)[-1]
            tc2 = tf.nn.dropout(tc2,
                               keep_prob=1.0,
                               name='tc2d',
                               noise_shape=[tf.shape(self.s_)[0], 1, 1, channels])

            tc2 = tf.reshape(tc2,
                        shape=[tf.shape(self.s)[0],
                               self.split_size,
                               self.window_size*20])


            gru_cells_ = []
            for i in range(2):
                cell = tf.nn.rnn_cell.GRUCell(
                    kernel_initializer=w_initializer,
                    num_units=128,
                    name="tg%d"%i)
                #GRU DROPOUT
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell,
                    output_keep_prob=1.0,
                    variational_recurrent=True,
                    dtype=tf.float32)
                gru_cells_.append(cell)

            multicell_ = tf.nn.rnn_cell.MultiRNNCell(gru_cells_)
            output, final_state = tf.nn.dynamic_rnn(
                cell=multicell_,
                inputs=tc2,
                dtype=tf.float32
            )
            output = tf.unstack(output, axis=1)[-1]

            t1 = tf.layers.dense(output, 128, tf.nn.relu,
                                 kernel_initializer=w_initializer, name='t1')
            t1 = tf.nn.dropout(t1, keep_prob=1.0, name='t1d')


            t2 = tf.layers.dense(t1, 64, tf.nn.relu,
                                 kernel_initializer=w_initializer, name='t2')
            t2 = tf.nn.dropout(t2, keep_prob=1.0, name='t2d')

            t3 = tf.layers.dense(t2, 32, tf.nn.relu,
                                 kernel_initializer=w_initializer, name='t3')
            t3 = tf.nn.dropout(t3, keep_prob=1.0, name='t3d')

            self.q_next = tf.layers.dense(t3, self.n_actions,
                                          kernel_initializer=w_initializer, name='t4')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.learning_rate, momentum=0.95, epsilon=0.01).minimize(self.loss)

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