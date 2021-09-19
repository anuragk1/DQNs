import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import deque
import copy
import time
import math, random

from torch.nn.modules.module import Module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

timesteps = 40000
batch_size = 512
network_hidden = 128
gamma = 0.95
lr=3*1e-5
max_buffer_size = timesteps // 2
min_buffer_size = batch_size // 2
max_traj_length = 500

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 2000

epsilon_on_episode = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

class Model(nn.Module):
    def __init__(self, num_input, num_output, network_hidden=network_hidden):
        super(Model, self).__init__()
        self.num_input = num_input
        self.num_output = num_output

        self.feature_layer = (
            nn.Sequential(
                nn.Linear(num_input, network_hidden),
                nn.ReLU()
            )
            .to(device)
            .to(dtype)
        )

        self.advantage_layer = (
            nn.Sequential(
                nn.Linear(network_hidden, network_hidden),
                nn.ReLU(),
                nn.Linear(network_hidden, num_output)  
            )
            .to(device)
            .to(dtype)
        )

        self.value_layer = (
            nn.Sequential(
               nn.Linear(network_hidden, network_hidden),
                nn.ReLU(),
                nn.Linear(network_hidden, 1)  
            )
            .to(device)
            .to(dtype)
        )
    
    def forward(self, x):
        x = self.feature_layer(x)
        advantage = self.advantage_layer(x)
        value     = self.value_layer(x)

        return value + advantage - advantage.mean()

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.num_output)
            # return self.env.action_space.sample()
        with torch.no_grad():
            state = torch.tensor(state, device=device, dtype=dtype).unsqueeze(0)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].item()
        return action


class DuelingDQN:
    def __init__(self, env, network_hidden=network_hidden, max_buffer_size=max_buffer_size, lr=lr):
        self.env = env
        self.network_hidden = network_hidden
        if self.env.unwrapped.spec is not None:
            self.env_name = self.env.unwrapped.spec.id
        else:
            self.env_name = self.env.unwrapped.__class__.__name__

        self.Q = Model(env.observation_space.shape[0], env.action_space.n)
        self.target_Q = copy.deepcopy(self.Q)

        self.replay_buffer = ReplayBuffer(max_buffer_size)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)

    # def create_model(self):
    #     self.feature_layer = (
    #         nn.Sequential(
    #             nn.Linear(self.env.observation_space.shape[0], network_hidden),
    #             nn.ReLU()
    #         )
    #         .to(device)
    #         .to(dtype)
    #     )

    #     self.advantage_layer = (
    #         nn.Sequential(
    #             nn.Linear(network_hidden, network_hidden),
    #             nn.ReLU(),
    #             nn.Linear(network_hidden, self.env.action_space.n)  
    #         )
    #         .to(device)
    #         .to(dtype)
    #     )

    #     self.value_layer = (
    #         nn.Sequential(
    #            nn.Linear(network_hidden, network_hidden),
    #             nn.ReLU(),
    #             nn.Linear(network_hidden, 1)  
    #         )
    #         .to(device)
    #         .to(dtype)
    #     )

    # def forward(self, x):
    #     x = self.Q.feature_layer(x)
    #     advantage = self.Q.advantage_layer(x)
    #     value     = self.Q.value_layer(x)

    #     return value + advantage - advantage.mean()

    # def select_action(self, state, epsilon):
    #     if random.random() < epsilon:
    #         # return random.randrange(self.env.action_space.n)
    #         return self.env.action_space.sample()
    #     with torch.no_grad():
    #         state = torch.tensor(state, device=device, dtype=dtype).unsqueeze(0)
    #         q_value = self.forward(state)
    #         action  = q_value.max(1)[1].item()
    #     return action

    def update(self, batch_size=batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.tensor(state, device=device, dtype=dtype)
        next_state = torch.tensor(next_state, device=device, dtype=dtype)
        action = torch.tensor(action, device=device, dtype=torch.long)
        reward = torch.tensor(reward, device=device, dtype=dtype)
        done = torch.tensor(done, device=device, dtype=dtype)

        q_values = self.Q(state)
        next_q_values = self.target_Q(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss = nn.MSELoss()(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, PLOT_REWARDS=False, VERBOSE=True, SAVE_FREQUENCY=None):
        print(f"\nTraining model on {self.env_name} | "f"Observation Space: {self.env.observation_space} | " f"Action Space: {self.env.action_space}\n")
        losses = []
        all_rewards = []

        step_count = 0

        if VERBOSE:
            print("Collecting experience ...")

        for episode in count():
            state = self.env.reset()
            episode_reward = []

            for _ in range(max_traj_length):
                step_count += 1
                epsilon = epsilon_on_episode(step_count // 4)
                action = self.Q.select_action(state, epsilon)
                next_state, reward, done, info_ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward.append(float(reward))

                if len(self.replay_buffer) > batch_size: 
                    loss = self.update(batch_size)
                    losses.append(loss)

                if SAVE_FREQUENCY is not None:
                    if (
                        len(self.replay_buffer) >= min_buffer_size
                        and step_count % (timesteps // SAVE_FREQUENCY) == 0
                    ):
                        self.save()

                if step_count % 100 == 0:
                    self.target_Q.load_state_dict(self.Q.state_dict())
                
                if done or step_count == timesteps:
                    break

            if step_count == timesteps:
                break

            total_episode_reward = sum(episode_reward)
            all_rewards.append(total_episode_reward)

            if VERBOSE:
                print(
                    f"Episode {episode+1}: Step Count = {step_count} | Reward = {total_episode_reward:.2f} | ",
                    end="",
                )
                if len(self.replay_buffer) >= batch_size:
                    print(f" DNQ Loss = {loss:.2f}")
                else:
                    print("Collecting Experience")
        if PLOT_REWARDS:
            plt.plot(all_rewards)
            plt.title(f"Training {self.__class__.__name__} on {self.env_name}")
            plt.xlabel("Episodes")
            plt.ylabel("Rewards")
            plt.savefig(f"./plots/{self.__class__.__name__}_{self.env_name}_reward_plot.png")

    def save(self, path=None):
        if path is None:
            path = f"./models/{self.__class__.__name__}_{self.env_name}.pt"
        torch.save({f"dqn_state_dict": self.Q.state_dict()}, path)
        print(f"\nSaved model parameters to {path}")

    def load(self, path=None):
        if path is None:
            path = f"./models/{self.__class__.__name__}_{self.env_name}.pt"
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint["dqn_state_dict"])
        print(f"\nLoaded model parameters from {path}")  

    def eval(self, episodes, RENDER=False):
        print(f"\nEvaluating model for {episodes} episodes ...\n")
        start_time = time.time()
        self.Q.eval()
        rewards = []

        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_rewards = []

            while not done:
                if RENDER:
                    self.env.render()
                
                action = self.Q.select_action(state, 0)
                next_state, reward, done, info_ = self.env.step(action)
                state = next_state
                episode_rewards.append(reward)

            total_episode_reward = sum(episode_rewards)
            rewards.append(total_episode_reward)
            print(f"Episode {episode+1}: Total Episode Reward = {total_episode_reward:.2f}")

        self.env.close()
        print(f"\nAverage Reward for an episode = {np.mean(rewards):.2f}")
        print(f"Evaluation Completed in {(time.time() - start_time):.2f} seconds\n")

if __name__ == "__main__":
    import gym
    env = gym.make('CartPole-v1')

    duelingdqn = DuelingDQN(env)
    duelingdqn.train(PLOT_REWARDS=True, SAVE_FREQUENCY=10)
    duelingdqn.save()
    duelingdqn.eval(10, RENDER=False)
        