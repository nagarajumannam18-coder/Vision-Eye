import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_dim=8, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class SignalControllerAgent:
    def __init__(self, model_path="models/traffic_dqn.pth"):
        self.model = DQN()
        self.lanes = ["North", "South", "East", "West"]
        try:
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.eval()
            self.is_trained = True
        except FileNotFoundError:
            self.is_trained = False

    def decide_signal(self, lane_data):
        """
        Returns: (selected_lane, green_duration_sec, reason)
        """
        # 1. Hard Priority Override for Emergency Vehicles
        for lane in self.lanes:
            if lane_data[lane]["emergency"]:
                return lane, 30, "EMERGENCY VEHICLE OVERRIDE"

        # Build state vector
        pcus = [lane_data[l]["pcu"] for l in self.lanes]
        emergs = [1.0 if lane_data[l]["emergency"] else 0.0 for l in self.lanes]
        state = torch.tensor(pcus + emergs, dtype=torch.float32).unsqueeze(0)

        # 2. RL Agent Action Selection (or fallback to max PCU if untrained)
        if self.is_trained:
            with torch.no_grad():
                q_values = self.model(state)
                action_idx = int(torch.argmax(q_values, dim=1).item())
        else:
            action_idx = int(np.argmax(pcus))

        chosen_lane = self.lanes[action_idx]
        chosen_pcu = pcus[action_idx]

        # 3. Dynamic Green Timer Calculation (Min 10s, Max 60s, ~2.5s per PCU)
        green_duration = int(np.clip(10 + (chosen_pcu * 2.5), 10, 60))
        mode = "DQN RL Agent" if self.is_trained else "Density Heuristic"
        return chosen_lane, green_duration, mode
