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
# room order clockwise: 0, 1, 2, 3
# hallway order clockwise: 0, 1, 2, 3
# 25, 56, 82, 103 are encoded hallway indices
NUM_ROOMS = 4
room_sizes = [[5, 5], [6, 5], [4, 5], [5, 5]]
options = get_options()  # get options for each room
terminal_hallways = get_terminal_hallways()
# primitive_actions = [UP, RIGHT, DOWN, LEFT]  # primitive action list
option_map = dict()  # maps option index to index in options
option_map[0] = [0, 1]
option_map[1] = [2, 3]
option_map[2] = [4, 5]
option_map[3] = [6, 7]
option_back_map = dict()
option_back_map[(0, 0)] = 0
option_back_map[(0, 1)] = 1
option_back_map[(1, 2)] = 0
option_back_map[(1, 3)] = 1
option_back_map[(2, 4)] = 0
option_back_map[(2, 5)] = 1
option_back_map[(3, 6)] = 0
option_back_map[(3, 7)] = 1
alpha = 0.125  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # epsilon for epsilon-greedy

# q_values = []
# testing code
offsets = env.offsets
# index = 56
# room = [r for r, offset in enumerate(offsets[1:5]) if index < offset][0]
# b = index in [offset - 1 for offset in offsets]
# print(room)
# print(b)


def init_q_values():
    q_values = []
    # initialize q values for each location including hallways
    for i in range(0, 104):
        q_list = list()
        for j in range(0, 6):  # first 2 for options, rest for primitive actions
            q_list.append(0)
        q_values.append(q_list)     # [opt1, opt2, UP, RIGHT, DOWN, LEFT]
    return q_values


n_iterations = 50
n_episodes = 10000

avg_rewards = []
avg_steps = []
tr = []
ts = []
for i in range(n_iterations):
    q_values = init_q_values()
    total_reward = []
    total_steps = []
    for j in range(n_episodes):
        print(i, j)
        option_over = True
        curr_option = 0
        opt_index = 0
        reward = 0  # total reward for episode
        steps = 0  # total steps in episode
        done = False
        g = 1  # gamma value
        curr_reward = 0  # reward obtained in previous step
        start_state = env.reset()
        curr_state = start_state
        while not done:
            if option_over:
                diff = curr_reward + g * np.max(q_values[curr_state]) - q_values[start_state][
                    opt_index]  # temporal difference
                q_values[start_state][opt_index] += alpha * diff  # update q value
                reward += curr_reward  # update total episode reward
                start_state = curr_state  # start state of option
                g = 1
                curr_reward = 0
                q_list = np.array(q_values[curr_state])
                x = np.random.rand()
                if x > epsilon:  # choose greedily
                    curr_option = np.argmax(q_list)
                else:  # choose randomly
                    curr_option = np.random.randint(0, 6)
                opt_index = curr_option  # index in q value list
                if curr_option < 2:  # if one of options
                    # print('Option running')
                    room = [r for r, offset in enumerate(offsets[1:5]) if curr_state < offset][0]  # current room
                    curr_option = option_map.get(room)[curr_option]  # get the index in options
                    # print(curr_option, curr_state)
                    if curr_state != terminal_hallways[curr_option]:
                        option_over = False
                    else:
                        # print('Primitive')
                        opt_list = []
                        for k in range(0, 6):
                            if k != opt_index:
                                opt_list.append(k)
                        idx = np.random.randint(0, 5)
                        curr_option = opt_list[idx]
                        opt_index = curr_option
                        if curr_option < 2:
                            room = [r for r, offset in enumerate(offsets[1:5]) if curr_state < offset][
                                0]  # current room
                            curr_option = option_map.get(room)[curr_option]  # get the index in options
                            option_over = False
            # print(curr_option)
            if not option_over:
                action = options[curr_option][curr_state]  # get action for current state in current option
            else:  # if primitive action
                action = curr_option - 2
            # print(curr_option, action)
            steps += 1
            new_state, rew, done, _ = env.step(action)
            curr_reward += g * rew
            g *= gamma
            # print(curr_state, new_state)
            if options[curr_option][new_state] != -1:
                curr_state = new_state
            else:  # new state not included in current option
                env.state = curr_state

            if not option_over:  # check if option is over
                option_over = (curr_state == terminal_hallways[curr_option])

        # print(start_state, opt_index)
        # update q value of terminal state
        diff = curr_reward + g * np.max(q_values[curr_state]) - q_values[start_state][
            opt_index]  # temporal difference
        q_values[start_state][opt_index] += alpha * diff  # update q value
        reward += curr_reward  # update total episode reward

        total_reward.append(reward)  # update total iteration reward
        total_steps.append(steps)  # update total iteration steps

    tr.append(total_reward)
    ts.append(total_steps)
    # avg_rewards.append(total_reward / n_episodes)
    # avg_steps.append(total_steps / n_episodes)

for i in range(n_episodes):
    tot = 0
    tot_steps = 0
    for j in range(n_iterations):
        tot += tr[j][i]
        tot_steps += ts[j][i]
    avg_rewards.append(tot / n_iterations)
    avg_steps.append(tot_steps / n_iterations)

# plotting graphs
# fig_rewards = plt.figure().add_subplot(111)
fig_steps = plt.figure().add_subplot(111)

# fig_rewards.set_xlabel('Iterations')
# fig_rewards.set_ylabel('Average Reward')
# fig_rewards.title.set_text('Average Reward vs Time')

fig_steps.set_xlabel('Episodes')
fig_steps.set_xscale('log')
fig_steps.set_yscale('log')
fig_steps.set_ylabel('Average Episode Length')
fig_steps.title.set_text('Average Episode Length vs Time')

# fig_rewards.plot(range(len(avg_rewards)), avg_rewards)
fig_steps.plot(range(len(avg_steps)), avg_steps)

plt.show()
