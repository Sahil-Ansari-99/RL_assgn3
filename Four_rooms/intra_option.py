import gym
import matplotlib.pyplot as plt
from options_generator import get_options, get_terminal_hallways
import numpy as np
import seaborn as sb

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
option_map = dict()  # maps option index to index in options
option_map[0] = [0, 1]
option_map[1] = [2, 3]
option_map[2] = [4, 5]
option_map[3] = [6, 7]
alpha = 0.125  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # epsilon for epsilon-greedy

offsets = env.offsets

n_iterations = 10
n_episodes = 10000

avg_rewards = []
avg_steps = []
tr = []
ts = []
last_q = np.zeros([104, 6])

for i in range(n_iterations):
    q_values = np.zeros([104, 6])
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
                # diff = curr_reward + gamma * np.max(q_values[curr_state]) - q_values[start_state][
                #     opt_index]  # temporal difference
                # q_values[start_state][opt_index] += alpha * diff  # update q value
                # reward += curr_reward  # update total episode reward
                # start_state = curr_state  # start state of option
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
            if not option_over:
                action = options[curr_option][curr_state]  # get action for current state in current option
            else:  # if primitive action
                action = curr_option - 2

            steps += 1
            new_state, rew, done, _ = env.step(action)
            curr_reward = rew
            reward += curr_reward

            fail = False  # check transition is valid or not
            if options[curr_option][new_state] != -1:
                start_state = curr_state
                curr_state = new_state
            else:  # new state not included in current option
                env.state = curr_state
                fail = True

            if not fail:
                # update q value of primitive action
                # action + 2 since action index starts from 2
                diff = curr_reward + gamma * np.max(q_values[curr_state]) - q_values[start_state][action + 2]
                q_values[start_state][action + 2] += alpha * diff

            if not option_over and not fail:  # execute only if an option was not selected
                beta = (curr_state == terminal_hallways[curr_option])  # is option terminated?
                if not beta:
                    diff = curr_reward + gamma * q_values[curr_state][opt_index] - q_values[start_state][opt_index]
                    q_values[start_state][opt_index] += alpha * diff  # update q value
                else:
                    diff = curr_reward + gamma * np.max(q_values[curr_state]) - q_values[start_state][opt_index]
                    q_values[start_state][opt_index] += alpha * diff

                room = [r for r, offset in enumerate(offsets[1:5]) if curr_state < offset][
                    0]  # current room
                if opt_index == 0:
                    other_opt = 1  # second option index
                else:
                    other_opt = 0
                other_idx = option_map.get(room)[other_opt]  # index of other option in option list
                if options[other_idx][start_state] == action:  # if other option action is the same
                    diff_2 = curr_reward + gamma * q_values[curr_state][other_opt] - q_values[start_state][other_opt]
                    q_values[start_state][other_opt] += alpha * diff_2  # update other option q value

                option_over = beta  # check if option is terminated

        total_reward.append(reward)  # update total iteration reward
        total_steps.append(steps)  # update total iteration steps

    tr.append(total_reward)
    ts.append(total_steps)
    last_q = q_values

# calculate average steps for each episode
for i in range(n_episodes):
    tot = 0
    tot_steps = 0
    for j in range(n_iterations):
        tot += tr[j][i]
        tot_steps += ts[j][i]
    avg_rewards.append(tot / n_iterations)
    avg_steps.append(tot_steps / n_iterations)

# plot q values
print(last_q)
q_map = np.zeros([13, 13])
q_map.fill(-0.5)  # default value for walls
starts = [0, 26, 57, 78]
dimens = [[5, 5], [6, 5], [4, 5], [5, 5]]
offs = [[1, 1], [1, 7], [8, 7], [7, 1]]
hallway = [[3, 6], [7, 9], [10, 6], [6, 2]]
cnt = 0
for i in range(0, 4):
    z = 0
    curr_start = starts[i]
    r = dimens[i][0]
    c = dimens[i][1]
    offs_r = offs[i][0]
    offs_c = offs[i][1]
    for j in range(r):
        for k in range(c):
            q_map[j+offs_r][k+offs_c] = np.max(last_q[cnt])
            if np.max(last_q[cnt]) == 0:
                z += 1
            cnt += 1
    q_map[hallway[i][0]][hallway[i][1]] = np.max(last_q[cnt])
    if np.max(last_q[cnt]) == 0:
        z += 1
    cnt += 1
    print(z)
ax = sb.heatmap(q_map, linewidths=0.5)

# plot steps and reward curve
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
