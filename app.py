import streamlit as st
import cv2
import numpy as np
import tempfile
from src.vision_detector import TrafficVisionDetector
from src.rl_agent import SignalControllerAgent

st.set_page_config(page_title="AI Smart Traffic Control", layout="wide")
st.title("🚦 Dynamic Traffic Signal Control (YOLOv11 + Deep RL)")

@st.cache_resource
def load_modules():
    return TrafficVisionDetector("yolo11n.pt"), SignalControllerAgent()

detector, agent = load_modules()

# Sidebar controls
st.sidebar.header("Simulation Controls")
mode = st.sidebar.radio("Input Source", ["Interactive Traffic Simulator", "Upload Intersection Video"])
emergency_lane = st.sidebar.selectbox("Simulate Ambulance Siren In:", ["None", "North", "South", "East", "West"])

if mode == "Interactive Traffic Simulator":
    st.subheader("Live Intersection Density & RL Signal Allocation")
    col1, col2, col3, col4 = st.columns(4)

    n_pcu = col1.slider("North Lane PCU Load", 0, 40, 18)
    s_pcu = col2.slider("South Lane PCU Load", 0, 40, 6)
    e_pcu = col3.slider("East Lane PCU Load", 0, 40, 28)
    w_pcu = col4.slider("West Lane PCU Load", 0, 40, 10)

    lane_data = {
        "North": {"count": int(n_pcu), "pcu": float(n_pcu), "emergency": emergency_lane == "North"},
        "South": {"count": int(s_pcu), "pcu": float(s_pcu), "emergency": emergency_lane == "South"},
        "East":  {"count": int(e_pcu), "pcu": float(e_pcu), "emergency": emergency_lane == "East"},
        "West":  {"count": int(w_pcu), "pcu": float(w_pcu), "emergency": emergency_lane == "West"},
    }

    active_lane, green_time, decision_source = agent.decide_signal(lane_data)

    st.divider()
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("🟢 Active Green Signal", f"{active_lane} Lane")
    res_col2.metric("⏱️ Allocated Green Duration", f"{green_time} sec", delta=f"{green_time - 30}s vs Fixed Timer")
    res_col3.metric("🧠 Decision Engine", decision_source)

    # Visual Signal Status Grid
    st.write("### Signal Phase Status")
    sig_cols = st.columns(4)
    for idx, lane in enumerate(["North", "South", "East", "West"]):
        is_green = (lane == active_lane)
        status = f"🟢 GREEN ({green_time}s)" if is_green else "🔴 RED (WAIT)"
        emerg_badge = " 🚑 **AMBULANCE**" if lane_data[lane]["emergency"] else ""
        sig_cols[idx].info(f"**{lane} Lane**{emerg_badge}\n\nLoad: {lane_data[lane]['pcu']} PCU\n\n{status}")

else:
    uploaded_file = st.file_uploader("Upload CCTV / Traffic Video (.mp4, .avi)", type=["mp4", "avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()
        metrics_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1100, 720))
            annotated_frame, lane_data = detector.process_frame(frame)

            if emergency_lane != "None":
                lane_data[emergency_lane]["emergency"] = True

            active_lane, green_time, source = agent.decide_signal(lane_data)

            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            with metrics_placeholder.container():
                st.success(f"**🟢 GREEN: {active_lane} Lane ({green_time}s)** | Engine: {source} | Counts: { {k: v['count'] for k, v in lane_data.items()} }")

        cap.release()