import gym
from torch import optim
import torch
import matplotlib.pyplot as plt

from utils.learn import e_greedy_action
# from utils.logger import Logger
from utils.models import ReplayMemory, History
from utils.net import DeepQNetwork, Q_targets, Q_values, save_network, copy_network, gradient_descent
from utils.processing import phi_map, tuple_to_numpy

import wimblepong

# ----------------------------------
# Tranining
env = gym.make("WimblepongVisualSimpleAI-v0")
env.unwrapped.scale = 1
env.unwrapped.fps = 30

def plot_rewards(rewards):
    plt.figure(1)
    plt.clf()
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative reward')
    plt.grid(True)
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

# Current iteration
step = 0
# Has trained model
has_trained_model = False
# Init training params
params = {
    'num_episodes': 5000,
    'minibatch_size': 32,
    'max_episode_length': int(10e6),  # T
    'memory_size': int(4.5e2),  # N
    'history_size': 4,  # k
    'train_freq': 4,
    'target_update_freq': 10,  # C: Target nerwork update frequency
    'num_actions': env.action_space.n,
    'min_steps_train': 50
}
# Initialize Logger
# log = Logger(log_dir="/log")
# Initialize replay memory D to capacity N
D = ReplayMemory(N=params['memory_size'],
                 load_existing=False, data_dir=".")
skip_fill_memory = D.count > 0
# Initialize action-value function Q with random weights
Q = DeepQNetwork(params['num_actions'])
# log.network(Q)
# Initialize target action-value function Q^
Q_ = copy_network(Q)
# Init network optimizer
optimizer = optim.RMSprop(
    Q.parameters(), lr=0.00025, alpha=0.95, eps=.01  # ,momentum=0.95,
)
# Initialize sequence s1 = {x1} and preprocessed sequence phi = phi(s1)
H = History.initial(env)
win = 0
cul_rewards = []

for ep in range(params['num_episodes']):
    print("Episode: {}, Wins: {}".format(ep, win))
    rewards = []

    phi = phi_map(H.get())
    # del phi

    if (ep % 10) == 0:
        save_network(Q, ep, out_dir="./data")

    for _ in range(params['max_episode_length']):
        # env.render(mode='human')
        # if step % 100 == 0:
        #     print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        step += 1
        # Select action a_t for current state s_t
        action, epsilon = e_greedy_action(Q, phi, env, step)
        # if step % FLAGS.log_freq == 0:
        #     log.epsilon(epsilon, step)
        # Execute action a_t in emulator and observe reward r_t and image x_(t+1)
        image, reward, done, _ = env.step(action)

        actual_reward = reward
        rewards = actual_reward

        # Clip reward to range [-1, 1]
        reward = max(-1.0, min(reward, 1.0))
        # Set s_(t+1) = s_t, a_t, x_(t+1) and preprocess phi_(t+1) =  phi_map( s_(t+1) )
        H.add(image)
        phi_prev, phi = phi, phi_map(H.get())
        # Store transition (phi_t, a_t, r_t, phi_(t+1)) in D
        D.add((phi_prev, action, reward, phi, done))

        should_train_model = skip_fill_memory or \
            ((step > params['min_steps_train']) and
             D.can_sample(params['minibatch_size']) and
             (step % params['train_freq'] == 0))

        if should_train_model:
            if not (skip_fill_memory or has_trained_model):
                D.save(params['min_steps_train'])
            has_trained_model = True

            # Sample random minibatch of transitions ( phi_j, a_j, r_j, phi_(j+1)) from D
            phi_mb, a_mb, r_mb, phi_plus1_mb, done_mb = D.sample(
                params['minibatch_size'])
            # Perform a gradient descent step on ( y_j - Q(phi_j, a_j) )^2
            y = Q_targets(phi_plus1_mb, r_mb, done_mb, Q_)
            q_values = Q_values(Q, phi_mb, a_mb)
            q_phi, loss = gradient_descent(y, q_values, optimizer)
            # Log Loss
            # if step % (params['train_freq'] * 1) == 0:
            #     # log.q_loss(q_phi, loss, step)
            # Reset Q_
            if step % params['target_update_freq'] == 0:
                del Q_
                Q_ = copy_network(Q)

        # log.episode(reward)
        # if FLAGS.log_console:
        #     log.display()


        # env.render()

        # # Restart game if done
        if done:
            H = History.initial(env)
            cul_rewards.append(rewards)
            if actual_reward == 10:
                win += 1
            # log.reset_episode()
            break

plot_rewards(cul_rewards)
plt.show()
