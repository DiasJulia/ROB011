# tiago_remote (v0.4)

Phone web remote for TIAGo (ROS Noetic). Two tabs:

- **Drive** — joystick → `/phone_vel` (twist_mux input). Max 1.0 m/s, 1.5 rad/s.
- **Translate** — live camera with face box, head arrows + "Find face", live captions.
  Signs are recognised only while a face is in view (no calibration step).

## Install / update

```bash
sudo apt install -y python3-tornado python3-opencv opencv-data unzip
cd ~/tiago_public_ws/src && unzip <path>/tiago_remote_v2.zip
cd ~/tiago_public_ws && catkin build tiago_remote && source devel/setup.bash
```

## Test in simulation (with a dummy person)

```bash
# terminal 1
roslaunch tiago_2dnav_gazebo tiago_navigation.launch public_sim:=true lost:=false end_effector:=pal-gripper
# terminal 2 (after Gazebo is up)
roslaunch tiago_remote sim_test.launch          # person_x:=2.0 person_y:=0.5 to move the person
```

Open http://localhost:8080. The dummy person can be moved in Gazebo with the "translate" tool (T).

## With the webcam + sign recogniser (real robot, or your PC)

```bash
roslaunch tiago_remote remote.launch                         # server (WSL / robot notebook)
python signvoice_robot.py --show                             # in the signvoice folder (Python 3.12 venv)
```

- phone: http://<PC-IP>:8080/   ·   robot screen: http://localhost:8080/screen.html (Full screen)
- `signvoice_robot.py` connects to `ws://localhost:8080/ws/recognizer` (use `--server` for another host)

## ROS interface

| Topic / action | Type | Dir | |
|---|---|---|---|
| `/phone_vel` | geometry_msgs/Twist | out | joystick |
| `/robot_mode` | std_msgs/String (latched) | out | drive / translate |
| `/translation_active` | std_msgs/Bool (latched) | out | translate mode **and** person in view → the sign recogniser should run only when true |
| `/head_controller/point_head_action` | control_msgs/PointHead | out | head arrows, find face |
| `/head_controller/command` | trajectory_msgs/JointTrajectory | out | head reset (dir "center") |
| `/person/face` | std_msgs/String JSON | in | `{"visible", "u", "v", "w", "h"}` normalised |
| `/sign/caption`, `/sign/glosses` | std_msgs/String | in | text shown in Translate |
