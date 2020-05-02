import tensorflow as tf
import numpy as np
from Model import Model


class DQN:
    def __init__(self, actions, input_size, hidden_layers, hidden_units):
        self.num_actions = actions
        self.replay_memory_size = 10000
        self.batch_size = 32
        self.gamma = 0.9  # discount factor
        self.learning_rate = 0.0001  # learning rate
        self.optimizer = tf.optimizers.SGD(self.learning_rate)
        self.model = Model(input_size, hidden_layers, hidden_units, actions)  # model for DQN
        self.replay_memory = []  # replay buffer for storing tuples

    def predict(self, x):
        return self.model(np.atleast_2d(x.astype('float32')))  # output for input x

    def train(self, target_network):
        if len(self.replay_memory) < self.batch_size:
            return  # don't train if insufficient samples
        ids = np.random.randint(0, len(self.replay_memory), size=self.batch_size)
        states = np.asarray([self.replay_memory[i][0] for i in ids])  # states
        actions = np.asarray([self.replay_memory[i][1] for i in ids])  # actions
        rewards = np.asarray([self.replay_memory[i][2] for i in ids])  # transition rewards
        next_states = np.asarray([self.replay_memory[i][3] for i in ids])  # next states
        is_done = np.asarray([self.replay_memory[i][4] for i in ids])  # was the episode terminated at this point?
        pred_q = np.max(target_network.predict(next_states), axis=1)
        target_values = np.where(is_done, rewards, rewards + self.gamma * pred_q)

        with tf.GradientTape() as g:
            # q values of action selected, returns [batch_size, 1]
            q_a = tf.reduce_sum(self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            diff = tf.reduce_mean(tf.square(target_values - q_a))  # sum of square of errors
        weights = self.model.trainable_variables
        grads = g.gradient(diff, weights)
        self.optimizer.apply_gradients(zip(grads, weights))
        return diff.numpy()

    def get_action(self, state, curr_epsilon):
        # epsilon-greedy
        if np.random.random() < curr_epsilon:
            return np.random.randint(self.num_actions)
        else:
            q = self.predict(np.atleast_2d(state))[0]
            return np.argmax(q)  # predict returns 2d list

    def update_replay_memory(self, exp, step_number):
        if len(self.replay_memory) < self.replay_memory_size:
            # if empty space in replay buffer
            self.replay_memory.append(exp)
        else:
            # override previous experiences
            self.replay_memory[step_number % self.replay_memory_size] = exp

