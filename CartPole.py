import gym
from dqn import DQN
import matplotlib.pyplot as plt
import numpy as np

def get_reward(observation, env):
    # the smaller theta and closer to center the better
    x, x_dot, theta, theta_dot = observation
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    return r1 + r2


# train model using DQN algorithm
def train(lr, e_greedy, times=100):
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    rlmodel = DQN(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        learning_rate=lr, e_greedy=e_greedy,
        replace_loop=100, memory_size=2000,
        e_greedy_increment=0.001,
        show_info=False,
    )
    total_steps = 0
    history = []
    for i_episode in range(times):
        observation = env.reset()
        ep_r = 0
        cur_steps = 0
        while True:
            action = rlmodel.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            reward = get_reward(observation, env)
            rlmodel.store_transition(observation, action, reward, observation_)
            ep_r += reward
            if total_steps > 1000:
                rlmodel.learn()
            if done or cur_steps >= 10000:
                history.append(ep_r)
                break
            observation = observation_
            cur_steps += 1
            total_steps += 1
    env.close()
    return rlmodel, history

# test model
def test(rlmodel, times=10):
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    history = []
    for i_episode in range(times):
        observation = env.reset()
        ep_r = 0
        cur_steps = 0
        while True:
            action = rlmodel.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            reward = get_reward(observation, env)
            ep_r += reward
            if done or cur_steps >= 10000:
                history.append(ep_r)
                break
            cur_steps += 1
            observation = observation_
    env.close()
    return history


def get_average_reward(lr, e_greedy):
    model, _ = train(lr, e_greedy, 100)
    history = test(model, 10)
    return np.mean(history)


avg = 0.0
for i in range(5):
    t = get_average_reward(0.0072, 0.98664545300265527)
    print(t)
    avg += t
avg /= 5
print(avg)

