import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from traffic_env import TrafficIntersectionEnv
from rl_agent import DQN

def train_dqn(episodes=400):
    env = TrafficIntersectionEnv()
    policy_net = DQN()
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    memory = deque(maxlen=5000)

    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.99
    batch_size = 64

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = int(torch.argmax(policy_net(torch.tensor(state))).item())

            next_state, reward, done, _, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_t = torch.tensor(states, dtype=torch.float32)
                actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards_t = torch.tensor(rewards, dtype=torch.float32)
                next_states_t = torch.tensor(next_states, dtype=torch.float32)
                dones_t = torch.tensor(dones, dtype=torch.float32)

                q_vals = policy_net(states_t).gather(1, actions_t).squeeze()
                next_q_vals = policy_net(next_states_t).max(1)[0]
                targets = rewards_t + gamma * next_q_vals * (1 - dones_t)

                loss = loss_fn(q_vals, targets.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes} | Total Reward: {total_reward:.1f} | Epsilon: {epsilon:.2f}")

    os.makedirs("models", exist_ok=True)
    torch.save(policy_net.state_dict(), "models/traffic_dqn.pth")
    print("Model saved to models/traffic_dqn.pth")

if __name__ == "__main__":
    train_dqn()
