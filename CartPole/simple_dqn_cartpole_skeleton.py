import gym
import random

import numpy as np
import tensorflow as tf

class DQN:

	REPLAY_MEMORY_SIZE = 10000 			# number of tuples in experience replay
	EPSILON = 0.5 						# epsilon of epsilon-greedy exploation
	EPSILON_DECAY = 0.99 				# exponential decay multiplier for epsilon
	HIDDEN1_SIZE = 128 					# size of hidden layer 1
	HIDDEN2_SIZE = 128 					# size of hidden layer 2
	EPISODES_NUM = 2000 				# number of episodes to train on. Ideally shouldn't take longer than 2000
	MAX_STEPS = 200 					# maximum number of steps in an episode
	LEARNING_RATE = 0.0001 				# learning rate and other parameters for SGD/RMSProp/Adam
	MINIBATCH_SIZE = 10 				# size of minibatch sampled from the experience replay
	DISCOUNT_FACTOR = 0.9 				# MDP's gamma
	TARGET_UPDATE_FREQ = 100 			# number of steps (not episodes) after which to update the target networks
	LOG_DIR = './logs' 					# directory wherein logging takes place


	# Create and initialize the environment
	def __init__(self, env):
		self.env = gym.make(env)
		assert len(self.env.observation_space.shape) == 1
		self.input_size = self.env.observation_space.shape[0]		# In case of cartpole, 4 state features
		self.output_size = self.env.action_space.n					# In case of cartpole, 2 actions (right/left)
		self.HIDDEN1_SIZE = 256  # size of hidden layer 1
		self.HIDDEN2_SIZE = 256
		self.num_actions = 2
		self.epsilon_freq = 20
		self.min_epsilon = 0.05
		self.LAMBDA = 0.001

# Create the Q-network
	def initialize_network(self):
		tf.compat.v1.disable_eager_execution()
		# placeholder for the state-space input to the q-network
		self.x = tf.compat.v1.placeholder(tf.float32, [None, self.input_size])

		############################################################
		# Design your q-network here.
		#
		# Add hidden layers and the output layer. For instance:
		#
		# with tf.name_scope('output'):
		#	W_n = tf.Variable(
		# 			 tf.truncated_normal([self.HIDDEN_n-1_SIZE, self.output_size],
		# 			 stddev=0.01), name='W_n')
		# 	b_n = tf.Variable(tf.zeros(self.output_size), name='b_n')
		# 	self.Q = tf.matmul(h_n-1, W_n) + b_n
		#
		#############################################################

		# Your code here
		self.weights = {
			'w1': tf.Variable(tf.random.truncated_normal([self.input_size, self.HIDDEN1_SIZE], stddev = 0.01), dtype=tf.float32, name='hidden1'),
			'b1': tf.Variable(tf.zeros(self.HIDDEN1_SIZE), dtype=tf.float32, name='bias1'),
			'w2': tf.Variable(tf.random.truncated_normal([self.HIDDEN1_SIZE, self.HIDDEN2_SIZE], stddev=0.01), dtype=tf.float32, name='hidden2'),
			'b2': tf.Variable(tf.zeros(self.HIDDEN2_SIZE), dtype=tf.float32, name='bias2'),
			'w_output': tf.Variable(tf.random.truncated_normal([self.HIDDEN2_SIZE, self.output_size], stddev=0.01), dtype=tf.float32, name='output'),
			'b_output': tf.Variable(tf.zeros(self.output_size), dtype=tf.float32, name='bias_output')
		}

		self.target_network = {
			'w1': tf.Variable(tf.random.truncated_normal([self.input_size, self.HIDDEN1_SIZE], stddev=0.01),
							  dtype=tf.float32, name='target_hidden1'),
			'b1': tf.Variable(tf.zeros(self.HIDDEN1_SIZE), dtype=tf.float32, name='target_bias1'),
			'w2': tf.Variable(tf.random.truncated_normal([self.HIDDEN1_SIZE, self.HIDDEN2_SIZE], stddev=0.01),
							  dtype=tf.float32, name='target_hidden2'),
			'b2': tf.Variable(tf.zeros(self.HIDDEN2_SIZE), dtype=tf.float32, name='target_bias2'),
			'w_output': tf.Variable(tf.random.truncated_normal([self.HIDDEN2_SIZE, self.output_size], stddev=0.01),
									dtype=tf.float32, name='target_output'),
			'b_output': tf.Variable(tf.zeros(self.output_size), dtype=tf.float32, name='target_bias_output')
		}

		h1_output = tf.linalg.matmul(self.x, self.weights['w1']) + self.weights['b1']
		h1_output = tf.nn.relu(h1_output)
		h2_output = tf.linalg.matmul(h1_output, self.weights['w2'] + self.weights['b2'])
		h2_output = tf.nn.relu(h2_output)
		self.Q = tf.linalg.matmul(h2_output, self.weights['w_output']) + self.weights['b_output']

		h1_output_t = tf.linalg.matmul(self.x, self.target_network['w1']) + self.target_network['b1']
		h1_output_t = tf.nn.relu(h1_output_t)
		h2_output_t = tf.linalg.matmul(h1_output_t, self.target_network['w2'] + self.target_network['b2'])
		h2_output_t = tf.nn.relu(h2_output_t)
		self.Q_t = tf.linalg.matmul(h2_output_t, self.target_network['w_output']) + self.target_network['b_output']

		############################################################
		# Next, compute the loss.
		#
		# First, compute the q-values. Note that you need to calculate these
		# for the actions in the (s,a,s',r) tuples from the experience replay's minibatch
		#
		# Next, compute the l2 loss between these estimated q-values and
		# the target (which is computed using the frozen target network)
		#
		############################################################

		# Your code here
		self.curr_a = tf.compat.v1.placeholder(tf.int32, [None], name='curr_a')
		self.one_hot_vector = tf.one_hot(self.curr_a, self.output_size)
		self.Q_val = tf.reduce_sum(tf.multiply(self.Q, self.one_hot_vector, name='one_hot_q'), axis=1)

		self.target = tf.compat.v1.placeholder(tf.float32, [None], name='target')
		self.error = self.target - self.Q_val
		self.loss = (tf.reduce_mean(tf.square(self.error)) + self.LAMBDA * (tf.nn.l2_loss(self.weights['w1']) + tf.nn.l2_loss(self.weights['w2']) + tf.nn.l2_loss(self.weights['w_output'])))

		############################################################
		# Finally, choose a gradient descent algorithm : SGD/RMSProp/Adam.
		#
		# For instance:
		# optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
		# global_step = tf.Variable(0, name='global_step', trainable=False)
		# self.train_op = optimizer.minimize(self.loss, global_step=global_step)
		#
		############################################################

		# Your code here
		optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.LEARNING_RATE)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		self.train_op = optimizer.minimize(self.loss, global_step=global_step)

		############################################################

	def train(self, episodes_num=EPISODES_NUM):

		# Initialize summary for TensorBoard
		summary_writer = tf.compat.v1.summary.FileWriter(self.LOG_DIR)
		summary = tf.compat.v1.Summary()
		# Alternatively, you could use animated real-time plots from matplotlib
		# (https://stackoverflow.com/a/24228275/3284912)

		# Initialize the TF session
		self.session = tf.compat.v1.Session()
		self.session.run(tf.compat.v1.global_variables_initializer())

		############################################################
		# Initialize other variables (like the replay memory)
		############################################################

		# Your code here
		replay_memory = []
		iteration_number = 0
		running_avg = []
		total_reward = np.empty(self.EPISODES_NUM)

		############################################################
		# Main training loop
		#
		# In each episode,
		#	pick the action for the given state,
		#	perform a 'step' in the environment to get the reward and next state,
		#	update the replay buffer,
		#	sample a random minibatch from the replay buffer,
		# 	perform Q-learning,
		#	update the target network, if required.
		#
		#
		#
		# You'll need to write code in various places in the following skeleton
		#
		############################################################

		for episode in range(episodes_num):

			state = self.env.reset()

			############################################################
			# Episode-specific initializations go here.
			############################################################
			#
			# Your code here
			#
			eps_steps = 0
			eps_reward = 0
			eps_loss = 0
			############################################################

			while True:

				############################################################
				# Pick the next action using epsilon greedy and and execute it
				############################################################

				# Your code here
				action = self.get_action(state)
				if iteration_number % self.epsilon_freq == 0:
					self.EPSILON = max(self.min_epsilon, self.EPSILON * self.EPSILON_DECAY)
				############################################################
				# Step in the environment. Something like:
				# next_state, reward, done, _ = self.env.step(action)
				############################################################

				# Your code here
				next_state, reward, done, _ = self.env.step(action)
				eps_reward += reward
				eps_steps += 1
				iteration_number += 1

				############################################################
				# Update the (limited) replay buffer.
				#
				# Note : when the replay buffer is full, you'll need to
				# remove an entry to accommodate a new one.
				############################################################

				# Your code here
				curr_exp = [state, action, reward, next_state, done]
				if len(replay_memory) > self.REPLAY_MEMORY_SIZE:
					replay_memory[iteration_number % self.REPLAY_MEMORY_SIZE] = curr_exp.copy()
				else:
					replay_memory.append(curr_exp.copy())
				############################################################
				# Sample a random minibatch and perform Q-learning (fetch max Q at s')
				#
				# Remember, the target (r + gamma * max Q) is computed
				# with the help of the target network.
				# Compute this target and pass it to the network for computing
				# and minimizing the loss with the current estimates
				#
				############################################################

				# Your code here
				if len(replay_memory) > self.MINIBATCH_SIZE:
					ids = np.random.randint(0, len(replay_memory), size=self.MINIBATCH_SIZE)
					states = np.asarray([replay_memory[k][0] for k in ids])  # states
					actions = np.asarray([replay_memory[k][1] for k in ids])  # actions
					rewards = np.asarray([replay_memory[k][2] for k in ids])  # transition rewards
					next_states = np.asarray([replay_memory[k][3] for k in ids])  # next states
					is_done = np.asarray([replay_memory[k][4] for k in ids])  # was the episode terminated at this point?
					pred_q = np.max(self.session.run(self.Q_t, feed_dict={self.x: next_states}), axis=1)
					y_i = np.where(is_done, rewards, rewards + self.DISCOUNT_FACTOR * pred_q)

					_, cost = self.session.run([self.train_op, self.loss], feed_dict={self.x: states, self.target: y_i, self.curr_a: actions})
					eps_loss += cost

				############################################################
			  	# Update target weights.
			  	#
			  	# Something along the lines of:
				# if total_steps % self.TARGET_UPDATE_FREQ == 0:
				# 	target_weights = self.session.run(self.weights)
				############################################################

				# Your code here
				if iteration_number % self.TARGET_UPDATE_FREQ == 0:
					for key in self.weights:
						self.session.run(tf.compat.v1.assign(self.target_network[key], tf.identity(self.weights[key])))

				############################################################
				# Break out of the loop if the episode ends
				#
				# Something like:
				# if done or (episode_length == self.MAX_STEPS):
				# 	break
				#
				############################################################

				# Your code here
				state = next_state
				if done or eps_steps == self.MAX_STEPS:
					break

			############################################################
			# Logging.
			#
			# Very important. This is what gives an idea of how good the current
			# experiment is, and if one should terminate and re-run with new parameters
			# The earlier you learn how to read and visualize experiment logs quickly,
			# the faster you'll be able to prototype and learn.
			#
			# Use any debugging information you think you need.
			# For instance :
			total_reward[episode] = eps_reward
			avg_steps = total_reward[max(0, episode - 100): episode+1].mean()
			running_avg.append(avg_steps)

			print("Training: Episode = %d, Length = %d, Global step = %d" % (episode, eps_steps, eps_steps))
			summary.value.add(tag="episode length", simple_value=eps_steps)
			summary.value.add(tag='Running Average', simple_value=avg_steps)
			summary_writer.add_summary(summary, episode)

	def get_action(self, state):
		if np.random.random() < self.EPSILON:
			return np.random.randint(self.num_actions)
		else:
			q = self.session.run(self.Q, feed_dict={self.x: state.reshape(1, state.shape[0])})
			return q.argmax()

	# Simple function to visually 'test' a policy
	def playPolicy(self):

		done = False
		steps = 0
		state = self.env.reset()

		# we assume the CartPole task to be solved if the pole remains upright for 200 steps
		while not done and steps < 200:
			self.env.render()
			q_vals = self.session.run(self.Q, feed_dict={self.x: [state]})
			action = q_vals.argmax()
			state, _, done, _ = self.env.step(action)
			steps += 1

		return steps


if __name__ == '__main__':

	# Create and initialize the model
	dqn = DQN('CartPole-v0')
	dqn.initialize_network()

	print("\nStarting training...\n")
	dqn.train()
	print("\nFinished training...\nCheck out some demonstrations\n")

	# Visualize the learned behaviour for a few episodes
	results = []
	for i in range(50):
		episode_length = dqn.playPolicy()
		print("Test steps = ", episode_length)
		results.append(episode_length)
	print("Mean steps = ", sum(results) / len(results))

	print("\nFinished.")
	print("\nCiao, and hasta la vista...\n")