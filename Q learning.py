import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import cv2


env = gym.make("MountainCar-v0")
env.reset()
LEARNING_RATE = 0.01 # ranging from 0 to 1
DISCOUNT = 0.95 # Weight, It's a measure of how important do we find out future actions
EPISODES = 25000
SHOW_EVERY = 2000 # Let us know you're alive after every 2000 episodes

'''print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)'''

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
# Discrete observation space size, seperating the range into 20 buckets( discrete values)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
# Bucket Size

epsilon = 0.5
#The higher the epsilon, the more likely we are to just perform a random action and exploratory. Ranging from 0 to 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2 # We'll have an int instead of integer
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

'''Creating Q tables: 
    The first row will be the number of actions available in the Q table.
    The first column will be the number of states available in the Q table. It's kinda a combination 
    of for example position and velocity. For a given situation they go back to the Q table, check the q values for 
    a specific combination and choose the largest number assigned to the regarding action. Initially the agent is going 
    to do a lot of exploration.Over time it reaches a reward and then by using Q function it will back propagate that 
    reward to make for higher Q values for the action chained togather lead to reward'''

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
# size = every combination of observations + number of actions (which creates the dimension)
# print(q_table.shape)
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
def get_discrete_state(state):
    discrete_os_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.asytype(np.int))

for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())
    '''To get the max Q value:
    np.argmax(q_table[discrete_state])'''
    done = False
    while not done:
        # Deciding whether to use epsilon
        #np.random.random creates a random float between 0 and 1
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        '''if it gets ti the flag without any exploration or randomness it's just gonna quickly reach the goal though 
        there might be a more efficient way of getting there.'''
        new_state, reward, done = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        # print(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state +(action, )]
            new_q = (1 - (LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q))
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print(f'We made it on episode {episode}')
            # reward for completing things:
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f'Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])}')
        '''if episode % 10 == 0:
            np.save(f"qtables/{episode}-qtable.npy", q_table)'''

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='Avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='Min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='Max')
plt.legend(loc=4)
plt.show()

style.use('ggplot')


def get_q_color(value, vals):
    if value == max(vals):
        return "green", 1.0
    else:
        return "red", 0.3


fig = plt.figure(figsize=(12, 9))


for i in range(0, 25000, 10):
    print(i)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    q_table = np.load(f"qtables/{i}-qtable.npy")

    for x, x_vals in enumerate(q_table):
        for y, y_vals in enumerate(x_vals):
            ax1.scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
            ax2.scatter(x, y, c=get_q_color(y_vals[1], y_vals)[0], marker="o", alpha=get_q_color(y_vals[1], y_vals)[1])
            ax3.scatter(x, y, c=get_q_color(y_vals[2], y_vals)[0], marker="o", alpha=get_q_color(y_vals[2], y_vals)[1])

            ax1.set_ylabel("Action 0")
            ax2.set_ylabel("Action 1")
            ax3.set_ylabel("Action 2")

    # plt.show()
    plt.savefig(f"qtable_charts/{i}.png")
    plt.clf()


def make_video():
    # windows:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('qlearn.avi', fourcc, 60.0, (1200, 900))

    for i in range(0, 14000, 10):
        img_path = f"qtable_charts/{i}.png"
        print(img_path)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()


make_video()
