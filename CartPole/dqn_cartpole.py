import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import gym
import random


class DQN:
    def __init__(self, env_id):
        self.env = gym.make(env_id)
        self.replay_memory_size = 10000
        self.mini_batch_size = 32
        self.num_actions = self.env.action_space.n
        self.input_size = len(self.env.observation_space.sample())
        self.hidden_units = [128, 128]
        self.num_episodes = 2000
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.1
        self.learning_rate = 0.001
        self.gamma = 0.9
        self.update_freq = 100
        self.max_steps = 196
        self.logs_dir = './logs'
        self.step_number = 0
        self.replay_memory = []
        self.optimizer = Adam(self.learning_rate)
        self.train_network = self.initialize_network()
        self.target_network = self.initialize_network()

    def initialize_network(self):
        model = Sequential()
        model.add(Dense(256, input_shape=(self.input_size, ), activation='tanh'))
        model.add(Dense(256, activation='tanh'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(optimizer=Adam(self.learning_rate), loss='mse')
        return model

    def add_experience(self, exp, step_number):
        if len(self.replay_memory) < self.replay_memory_size:
            self.replay_memory.append(exp)
        else:
            self.replay_memory[step_number % self.replay_memory_size] = exp

    def reshape_input(self, x):
        return np.reshape(x, [1, len(x)])  # reshapes into 2d array

    def predict(self, x, network):
        x = self.reshape_input(x)
        return network.predict(x)  # q values for input x

    def batch_predict(self, x, network):
        return network.predict(x)

    def get_action(self, x):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)  # random action
        else:
            q_val = self.predict(x, self.train_network)[0]  # greedy action
            return np.argmax(q_val)

    def update_target_net(self):
        self.target_network.set_weights(self.train_network.get_weights())

    def train(self):
        if len(self.replay_memory) < self.mini_batch_size:
            return
        mini_batch = random.sample(self.replay_memory, self.mini_batch_size)
        x_batch = []
        y_batch = []
        for state, action, reward, new_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.predict(new_state, self.target_network)[0])
            curr = self.predict(state, self.train_network)
            curr[0][action] = target
            x_batch.append(state)
            y_batch.append(curr[0])
            # self.train_network.fit(self.reshape_input(state), curr, verbose=0, epochs=1)
        self.train_network.fit(np.array(x_batch), np.array(y_batch), verbose=0, batch_size=self.mini_batch_size)

    def gradient_train(self):
        if len(self.replay_memory) < self.mini_batch_size:
            return
        ids = np.random.randint(0, len(self.replay_memory), self.mini_batch_size)
        states = np.array([self.replay_memory[i][0] for i in ids])
        actions = np.array([self.replay_memory[i][1] for i in ids])
        rewards = np.array([self.replay_memory[i][2] for i in ids])
        next_states = np.array([self.replay_memory[i][3] for i in ids])
        terminated = np.array([self.replay_memory[i][4] for i in ids])
        curr_q = np.max(self.batch_predict(next_states, self.target_network), axis=1)
        target_values = np.where(terminated, rewards, rewards + self.gamma * curr_q)
        q = self.batch_predict(states, self.train_network)
        with tf.GradientTape() as tape:
            q_a = tf.reduce_sum(q * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.reduce_mean(tf.square(target_values - q_a))  # sum of square of errors
        weights = self.train_network.trainable_variables
        grads = tape.gradient(loss, weights)
        self.optimizer.apply_gradients(zip(grads, weights))

    def run_episode(self):
        rewards = 0
        steps = 0
        done = False
        state = self.env.reset()
        while not done:
            action = self.get_action(state)
            new_state, reward, done, _ = self.env.step(action)
            steps += 1
            rewards += reward
            self.step_number += 1
            curr_exp = [state, action, reward, new_state, done]
            # self.train()
            self.add_experience(curr_exp, self.step_number)
            # if self.step_number % self.update_freq == 0:
            #     self.update_target_net()
            if steps == self.max_steps:
                break
            state = new_state
        self.train()
        return steps, rewards

    def start_training(self):
        total_rewards = np.empty(self.num_episodes)
        total_steps = np.empty(self.num_episodes)
        avg_rewards = []
        consecutive_runs = 0
        summary_writer = tf.summary.create_file_writer(self.logs_dir)
        for i in range(self.num_episodes):
            if i % self.update_freq:
                self.update_target_net()
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            eps_reward, eps_steps = self.run_episode()
            total_rewards[i] = eps_reward
            total_steps[i] = eps_steps
            avg_reward = total_steps[max(0, i - 100): i + 1].mean()
            print(i, avg_reward)
            avg_rewards.append(avg_reward)
            if avg_reward > 190:
                consecutive_runs += 1
            with summary_writer.as_default():
                tf.summary.scalar('Running Average', avg_reward, i)
                tf.summary.scalar('Reward', eps_reward, i)
                tf.summary.scalar('Steps', eps_steps, i)
            if consecutive_runs >= 100:
                break
        return total_rewards, total_steps, avg_rewards

    def play_policy(self):
        done = False
        steps = 0
        state = self.env.reset()
        while not done and steps < 200:
            self.env.render()
            q_val = self.predict(state, self.train_network)[0]
            action = np.argmax(q_val)
            state, rew, done, _ = self.env.step(action)
            steps += 1
        print(steps)

    def plot_graphs(self, reward_list, step_list, avg_list):
        fig_rewards = plt.figure().add_subplot(111)
        fig_steps = plt.figure().add_subplot(111)
        fig_avg = plt.figure().add_subplot(111)

        fig_rewards.set_xlabel('Iterations')
        fig_rewards.set_ylabel('Reward')
        fig_rewards.title.set_text('Reward vs Time')

        fig_steps.set_xlabel('Episodes')
        fig_steps.set_ylabel('Episode Length')
        fig_steps.title.set_text('Episode Length vs Time')

        fig_avg.set_xlabel('Episodes')
        fig_avg.set_ylabel('Average Loss')
        fig_avg.title.set_text('Average Reward vs Episodes')

        fig_rewards.plot(range(len(reward_list)), reward_list)
        fig_steps.plot(range(len(step_list)), step_list)
        fig_avg.plot(range(len(avg_list)), avg_list)

        plt.show()


if __name__ == '__main__':
    dqn = DQN('CartPole-v0')
    rewards, steps, avg_steps = dqn.start_training()
    dqn.play_policy()
    dqn.plot_graphs(rewards, steps, avg_steps)
