# Dynamic Traffic Signal Control (YOLOv11 + Deep RL)

A real-time smart traffic signal management system using YOLOv11 for multi-lane vehicle detection (with Passenger Car Unit weighting) and a Deep Q-Network (DQN) Reinforcement Learning agent for dynamic green light allocation and emergency vehicle preemption.

## Project Structure
```text
smart_traffic_control/
├── data/                  # Place your intersection traffic video here (e.g., intersection_sample.mp4)
├── models/                # Saved PyTorch RL weights (traffic_dqn.pth)
├── src/
│   ├── __init__.py
│   ├── vision_detector.py # YOLOv11 vehicle & ambulance counter
│   ├── traffic_env.py     # Custom Gymnasium RL Environment
│   ├── rl_agent.py        # Deep Q-Network (DQN) Agent + Emergency Override
│   └── train.py           # Script to train the RL agent
├── app.py                 # Streamlit live dashboard & simulation UI
└── requirements.txt       # Project dependencies
```

## Setup & Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the Reinforcement Learning agent:
   ```bash
   python src/train.py
   ```
3. Launch the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```
