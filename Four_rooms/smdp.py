import gym
import matplotlib.pyplot as plt
from options_generator import get_options, get_terminal_hallways
import numpy as np

gym.register(id='FourRooms-v0',
                entry_point='envs.four_rooms:FourRooms')
env = gym.make('FourRooms-v0')

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# rooms and hallways are zero indexed
# room order clockwise: 0, 1, 3, 2
# hallway order clockwise: 0, 1, 2, 3
# 25, 56, 82, 103 are encoded hallway indices
NUM_ROOMS = 4
room_sizes = [[5, 5], [6, 5], [5, 5], [4, 5]]
options = get_options()  # get options for each room
terminal_hallways = get_terminal_hallways()
primitive_actions = [UP, RIGHT, DOWN, LEFT]  # primitive action list
option_map = dict()  # maps option index to index in options
option_map[0] = [0, 1]
option_map[1] = [2, 3]
option_map[2] = [4, 5]
option_map[3] = [6, 7]

alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # epsilon for epsilon-greedy

q_values = []
# testing code
offsets = [0, 26, 57, 83, 104]
index = 56
room = [r for r, offset in enumerate(offsets[1:5]) if index < offset][0]
b = index in [offset - 1 for offset in offsets]
print(room)
print(b)

# initialize q values for each location including hallways
for i in range(0, 104):
    q_list = list()
    for j in range(0, 6):  # first 2 for options, rest for primitive actions
        q_list.append(0)
    q_values.append(q_list)     # [opt1, opt2, UP, RIGHT, DOWN, LEFT]

n_iterations = 1000
n_episodes = 50

avg_rewards = []
avg_steps = []
for i in range(n_iterations):
    total_reward = 0
    total_steps = 0
    for j in range(n_episodes):
        option_over = True
        curr_option = 0
        reward = 0  # total reward for episode
        done = False
        g = 1  # gamma value
        curr_reward = 0 # reward obtained in previous step
        ob = env.reset()
        start_state = ob
        curr_state = start_state
        while not done:
            if option_over:
                diff = curr_reward + g * np.max(q_values[curr_state]) - q_values[start_state][curr_option]  # temporal difference
                q_values[start_state][curr_option] += alpha * diff  # update q value
                start_state = curr_state
                q_list = np.array(q_values[curr_state])
                x = np.random.rand()
                if x > epsilon:  # choose greedily
                    curr_option = np.argmax(q_list)
                else:  # choose randomly
                    curr_option = np.random.randint(0, 6)
                    if curr_option < 2:  # if one of options
                        room = [r for r, offset in enumerate(offsets[1:5]) if curr_state < offset][0]  # current room
                        curr_option = option_map.get(room)[curr_option]  # get the index in options
                        option_over = False

            if not option_over:
                action = options[curr_option][curr_state]
            else:
                action = curr_option - 2
            new_state, rew, done, _ = ob.step(action)
            curr_reward += g*rew
            g *= gamma
            curr_state = new_state

            if not option_over:
                option_over = (new_state == terminal_hallways[curr_option])
