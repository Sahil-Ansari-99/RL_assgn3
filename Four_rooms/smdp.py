import gym
import matplotlib.pyplot as plt
from options_generator import get_options, get_terminal_hallways
import numpy as np
import seaborn as sb


# rooms and hallways are zero indexed
# room order clockwise: 0, 1, 2, 3
# hallway order clockwise: 0, 1, 2, 3
# 25, 56, 82, 103 are encoded hallway indices
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

NUM_ROOMS = 4

gym.register(id='FourRooms-v0',
                entry_point='envs.four_rooms:FourRooms')


class SMDP:
    def __init__(self, id):
        self.env = gym.make(id)
        self.room_sizes = [[5, 5], [6, 5], [4, 5], [5, 5]]
        self.options = get_options()
        self.terminal_hallways = get_terminal_hallways()
        self.option_map = dict()  # get options for each room
        self.option_map[0] = [0, 1]
        self.option_map[1] = [2, 3]
        self.option_map[2] = [4, 5]
        self.option_map[3] = [6, 7]
        self.alpha = 0.125  # learning rate
        self.gamma = 0.9  # discount factor
        self.epsilon = 0.1  # epsilon for epsilon-greedy
        self.offsets = self.env.offsets
        self.n_iterations = 50
        self.n_episodes = 10000

    def train(self):
        tr = []
        ts = []
        last_q = np.zeros([104, 6])
        for i in range(self.n_iterations):
            q_values = np.zeros([104, 6])
            total_reward = []
            total_steps = []
            for j in range(self.n_episodes):
                print(i, j)
                option_over = True
                curr_option = 0
                opt_index = 0
                reward = 0  # total reward for episode
                steps = 0  # total steps in episode
                done = False
                g = 1  # gamma value
                curr_reward = 0  # reward obtained in previous step
                start_state = self.env.reset()
                curr_state = start_state
                while not done:
                    if option_over:
                        diff = curr_reward + g * np.max(q_values[curr_state]) - q_values[start_state][
                            opt_index]  # temporal difference
                        q_values[start_state][opt_index] += self.alpha * diff  # update q value
                        reward += curr_reward  # update total episode reward
                        start_state = curr_state  # start state of option
                        g = 1
                        curr_reward = 0
                        q_list = np.array(q_values[curr_state])
                        x = np.random.rand()
                        if x > self.epsilon:  # choose greedily
                            curr_option = np.argmax(q_list)
                        else:  # choose randomly
                            curr_option = np.random.randint(0, 6)
                        opt_index = curr_option  # index in q value list
                        if curr_option < 2:  # if one of options
                            # print('Option running')
                            room = [r for r, offset in enumerate(self.offsets[1:5]) if curr_state < offset][
                                0]  # current room
                            curr_option = self.option_map.get(room)[curr_option]  # get the index in options
                            # print(curr_option, curr_state)
                            if curr_state != self.terminal_hallways[curr_option]:
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
                                    room = [r for r, offset in enumerate(self.offsets[1:5]) if curr_state < offset][
                                        0]  # current room
                                    curr_option = self.option_map.get(room)[curr_option]  # get the index in options
                                    option_over = False
                    # print(curr_option)
                    if not option_over:
                        action = self.options[curr_option][curr_state]  # get action for current state in current option
                    else:  # if primitive action
                        action = curr_option - 2
                    # print(curr_option, action)
                    steps += 1
                    new_state, rew, done, _ = self.env.step(action)
                    curr_reward += g * rew
                    g *= self.gamma
                    # print(curr_state, new_state)
                    if self.options[curr_option][new_state] != -1:
                        curr_state = new_state
                    else:  # new state not included in current option
                        self.env.state = curr_state

                    if not option_over:  # check if option is over
                        option_over = (curr_state == self.terminal_hallways[curr_option])

                # print(start_state, opt_index)
                # update q value of terminal state
                if option_over:
                    diff = curr_reward + g * np.max(q_values[curr_state]) - q_values[start_state][
                        opt_index]  # temporal difference
                    q_values[start_state][opt_index] += self.alpha * diff  # update q value
                reward += curr_reward  # update total episode reward

                total_reward.append(reward)  # update total iteration reward
                total_steps.append(steps)  # update total iteration steps

            tr.append(total_reward)
            ts.append(total_steps)
            last_q = q_values
        self.plot_q_map(last_q)
        self.plot_graphs(tr, ts)

    # function to visualize q values
    def plot_q_map(self, q):
        print(q)
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
                    q_map[j + offs_r][k + offs_c] = np.max(q[cnt])
                    if np.max(q[cnt]) == 0:
                        z += 1
                    cnt += 1
            q_map[hallway[i][0]][hallway[i][1]] = np.max(q[cnt])
            if np.max(q[cnt]) == 0:
                z += 1
            cnt += 1
            print(z)
        ax = sb.heatmap(q_map, linewidths=0.5)

    # function to plot steps and reward curve
    def plot_graphs(self, tr, ts):
        avg_rewards = []
        avg_steps = []
        for i in range(self.n_episodes):
            tot = 0
            tot_steps = 0
            for j in range(self.n_iterations):
                tot += tr[j][i]
                tot_steps += ts[j][i]
            avg_rewards.append(tot / self.n_iterations)
            avg_steps.append(tot_steps / self.n_iterations)
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


if __name__ == '__main__':
    smdp = SMDP('FourRooms-v0')
    smdp.train()
