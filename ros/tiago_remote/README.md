# tiago_remote (v0.5)

> Full setup, robot connection and troubleshooting: [docs/TIAGO_GUIDE.md](../../docs/TIAGO_GUIDE.md)

Phone web remote for TIAGo (ROS Noetic), working together with `signvoice.py` (repository root).

- **Drive** — joystick → `/phone_vel` (twist_mux input). Max 1.0 m/s, 1.5 rad/s.
- **Conversation** — remote for the translator running on the robot's notebook: Talk / Listen,
  recognised signs (× to delete one), Undo, Clear, Speak, history (You / Them), LLM on/off, full screen.
  The robot's screen shows the original signvoice window (camera, signs, history).

## Install

```bash
sudo apt install -y python3-tornado
cp -r ros/tiago_remote ~/tiago_public_ws/src/      # or symlink it
cd ~/tiago_public_ws && catkin build tiago_remote && source devel/setup.bash
```

## Run

```bash
# 1. (optional) simulation
roslaunch tiago_2dnav_gazebo tiago_navigation.launch public_sim:=true lost:=false end_effector:=pal-gripper
# 2. web remote
roslaunch tiago_remote remote.launch
```

```bash
# 3. translator (repository root, Python 3.12 venv, pip install websocket-client)
python signvoice.py --server ws://127.0.0.1:8080/ws/translator --fullscreen
```

Open `http://localhost:8080` (PC) or `http://<PC-IP>:8080` (phone, same Wi-Fi).

With `--server`, the pointing gestures are off by default (ASL signs such as YOU / ME / THERE also point);
add `--gestures` to keep them. Keyboard and mouse still work on the notebook.

## ROS interface

| Topic | Type | Dir | |
|---|---|---|---|
| `/phone_vel` | geometry_msgs/Twist | out | joystick (published only while in use) |
| `/conversation/history` | std_msgs/String (latched) | out | JSON `[["you" \| "them", text], ...]` |
