import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TrafficIntersectionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 4 lane PCU densities + 4 emergency flags
        self.observation_space = spaces.Box(low=0, high=100, shape=(8,), dtype=np.float32)
        # 4 possible green light phases (N, S, E, W)
        self.action_space = spaces.Discrete(4)
        self.state = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Random initial traffic densities (0 to 30 PCUs per lane)
        densities = np.random.uniform(2, 25, size=(4,)).astype(np.float32)
        emergencies = np.random.choice([0.0, 1.0], size=(4,), p=[0.92, 0.08]).astype(np.float32)
        self.state = np.concatenate([densities, emergencies])
        self.steps = 0
        return self.state, {}

    def step(self, action):
        self.steps += 1
        densities = self.state[:4].copy()
        emergencies = self.state[4:].copy()

        # Discharge vehicles from the lane given the green light
        discharge_rate = np.random.uniform(6.0, 10.0)
        densities[action] = max(0.0, densities[action] - discharge_rate)
        emergencies[action] = 0.0  # Emergency vehicle clears the intersection

        # New incoming traffic arrives randomly
        arrivals = np.random.uniform(0.5, 3.5, size=(4,))
        densities = np.clip(densities + arrivals, 0, 100)

        # Calculate Reward: minimize total waiting congestion & prioritize emergencies
        waiting_penalty = -np.sum(densities)
        emergency_penalty = -100.0 * np.sum(emergencies)
        reward = float(waiting_penalty + emergency_penalty)

        # Randomly spawn new emergency vehicle occasionally
        new_emerg = np.random.choice([0.0, 1.0], size=(4,), p=[0.96, 0.04])
        emergencies = np.maximum(emergencies, new_emerg)

        self.state = np.concatenate([densities, emergencies]).astype(np.float32)
        terminated = self.steps >= 100
        return self.state, reward, terminated, False, {}
