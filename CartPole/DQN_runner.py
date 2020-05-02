import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
from DQN import DQN
import math


class DQNRunner:
    def __init__(self, id):
        self.env = gym.make(id)
        actions = self.env.action_space.n
        input_size = len(self.env.observation_space.sample())
        hidden_layers = 2
        hidden_units = [200, 200]
        self.episodes = 2000
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.curr_epsilon = 1.0
        self.min_epsilon = 0.05
        self.update_freq = 100
        self.iteration_number = 0
        self.epsilon_freq = 50
        self.logs_dir = './logs'
        self.train_network = DQN(actions, input_size, hidden_layers, hidden_units)
        self.target_network = DQN(actions, input_size, hidden_layers, hidden_units)

    def run_episode(self):
        reward = 0
        steps = 0
        done = False
        state = self.env.reset()
        eps_losses = []

        while not done:
            action = self.train_network.get_action(state, self.epsilon)  # get action from train network
            next_state, rew, done, _ = self.env.step(action)
            reward += rew  # update episode reward
            curr_exp = [state, action, rew, next_state, done]  # curr transition details
            self.train_network.update_replay_memory(curr_exp, self.iteration_number)
            self.iteration_number += 1
            steps += 1
            loss = self.train_network.train(self.target_network)  # update train network weights
            if loss is not None:
                eps_losses.append(loss)
            if self.iteration_number % self.update_freq == 0:
                # update target network weights
                self.update_target_net()
            state = next_state  # update state
            if steps > 195:
                break
        return reward, steps, np.mean(eps_losses)

    def update_target_net(self):
        # copy weights of train network to target network
        curr = self.train_network.model.get_weights()
        # new_vars = self.target_network.model.trainable_variables
        # for v1, v2 in zip(curr, new_vars):
        #     tf.compat.v1.assign(v1, v2)
        self.target_network.model.set_weights(curr)

    def train(self):
        total_rewards = np.empty(self.episodes)
        total_steps = []
        total_loss = []
        running_average = []
        summary_writer = tf.summary.create_file_writer(self.logs_dir)
        for i in range(self.episodes):
            self.epsilon  = min(self.min_epsilon, self.epsilon * self.epsilon_decay)
            eps_reward, eps_steps, eps_loss = self.run_episode()  # episode reward and average loss
            total_rewards[i] = eps_reward
            total_steps.append(eps_steps)
            total_loss.append(eps_loss)
            avg_reward = total_rewards[max(0, i - 100): i + 1].mean()
            print(i, avg_reward, self.epsilon)
            running_average.append(avg_reward)
            with summary_writer.as_default():
                tf.summary.scalar('Reward', eps_reward, i)
                tf.summary.scalar('Loss', eps_loss, i)
                tf.summary.scalar('Steps', eps_steps, i)
                tf.summary.scalar('Average Reward', avg_reward, i)
        return total_rewards, total_steps, total_loss

    def plot_graphs(self, rewards, steps, loss):
        fig_rewards = plt.figure().add_subplot(111)
        fig_steps = plt.figure().add_subplot(111)
        fig_loss = plt.figure().add_subplot(111)

        fig_rewards.set_xlabel('Iterations')
        fig_rewards.set_ylabel('Reward')
        fig_rewards.title.set_text('Reward vs Time')

        fig_steps.set_xlabel('Episodes')
        fig_steps.set_ylabel('Average Episode Length')
        fig_steps.title.set_text('Average Episode Length vs Time')

        fig_loss.set_xlabel('Episodes')
        fig_loss.set_ylabel('Average Loss')
        fig_loss.title.set_text('Average Loss vs Episodes')

        fig_rewards.plot(range(len(rewards)), rewards)
        fig_steps.plot(range(len(steps)), steps)
        fig_loss.plot(range(len(loss)), loss)

        plt.show()


if __name__ == '__main__':
    runner = DQNRunner('CartPole-v0')
    rewards, steps, loss = runner.train()
    # runner.plot_graphs(rewards, steps, loss)
