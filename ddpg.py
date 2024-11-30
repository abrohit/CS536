import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from env import NetworkEnv

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, gamma, tau, buffer_size, w_min, w_max):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = deque(maxlen=buffer_size)

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.w_min = w_min
        self.w_max = w_max

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).squeeze(0).detach().numpy()
        # action = action.numpy()
        action += np.random.normal(0, noise, size=action.shape)
        return np.clip(action, self.w_min, self.w_max)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train(self, batch_size):
        
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = random.sample(self.replay_buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # Update Critic
        next_action = self.actor_target(next_state)
        target_q = self.critic_target(next_state, next_action)
        target_q = reward + (1 - done) * self.gamma * target_q
        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Training loop
def train_ddpg(env, agent, num_episodes, max_steps, batch_size, topology):
    for episode in range(num_episodes):
        state, _ = env.reset()
        new_state = list()
        for key, value in state.items():
            new_state.append(value)
        state = np.concatenate(new_state)
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            #print("DONE STATUS: " + done)
            new_next_state = list()
            for key, value in next_state.items():
                new_next_state.append(value)

            next_state = np.concatenate(new_next_state)

            agent.store_transition(state, action, reward, next_state, done)
            agent.train(batch_size)

            state = next_state
            episode_reward += reward

            avg_delays = env.network.e_d.mean()
            avg_end_to_end_delay = env.network.d_k_e2e.mean()
            avg_loss = env.network.e_l.mean()
            # avg_throughput

            if done:
                break

            print(f"Step {step}, Reward: {reward}")
            with open("topology_" + str(topology) + "_report.txt", "a") as f:
                f.write(f"Step {step}, Reward: {reward}, Avg Delay: {avg_delays}, Avg E2E Delay: {avg_end_to_end_delay}, Avg Loss: {avg_loss}\n")

            if step % 10000 == 0:
                torch.save(agent, "models/agent_" + str(step) + "topology_" + str(topology) + ".pt")

# Main execution
if __name__ == "__main__":
    # Environment parameters
    topology = 0
    alpha = 0.9
    w_min = 1
    w_max = 5
    episode_length = 150000

    # Create environment
    env = NetworkEnv(topology, alpha, w_min, w_max, episode_length)

    # DDPG parameters
    state_dim = sum([space.shape[0] for space in env.observation_space.values()])
    action_dim = env.action_space.shape[0]
    hidden_dim = 256
    actor_lr = 1e-5
    critic_lr = 1e-5
    gamma = 0.99
    tau = 0.001
    buffer_size = 50000

    # Create DDPG agent
    agent = DDPGAgent(state_dim, action_dim, hidden_dim, actor_lr, critic_lr, gamma, tau, buffer_size, w_min, w_max)

    # Training parameters
    num_episodes = 1
    max_steps = episode_length
    batch_size = 100

    # Start training
    train_ddpg(env, agent, num_episodes, max_steps, batch_size, topology)
