# TIAGo + signvoice — full setup and run guide

This guide documents everything needed to run the project on the lab's TIAGo robot:
a phone drives the robot and controls the conversation, while `signvoice.py` translates
American Sign Language (ASL) and speech on the notebook that rides on the robot.

It was written from the steps that were actually done and tested, including the problems
we hit and how they were solved (see [Troubleshooting](#10-troubleshooting)).

**Contents**

1. [How it works](#1-how-it-works)
2. [The robot](#2-the-robot)
3. [Notebook setup (once)](#3-notebook-setup-once)
4. [Try it in simulation](#4-try-it-in-simulation)
5. [Power the robot on and off](#5-power-the-robot-on-and-off)
6. [Connect to the robot](#6-connect-to-the-robot)
7. [Run everything on the real robot](#7-run-everything-on-the-real-robot)
8. [Using the phone remote](#8-using-the-phone-remote)
9. [Safety checklist](#9-safety-checklist)
10. [Troubleshooting](#10-troubleshooting)
11. [Reference](#11-reference)

---

## 1. How it works

```
                     Wi-Fi (phone hotspot)                          Ethernet cable (10.68.0.x)
 Phone (browser) ───────────────────────────► Notebook on the robot ─────────────────────────► TIAGo (tiago-27c)
  Drive tab: joystick                           ├─ WSL Ubuntu 20.04 + ROS Noetic                 ROS Indigo, Ubuntu 14.04
  Conversation tab: Talk/Listen,                │    tiago_remote (web server, port 8080)        /phone_vel → twist_mux →
  Undo, Clear, Speak, ×, history,               │      ├─ /ws            ← phone                  speed_limit → wheels
  LLM, full screen                              │      └─ /ws/translator ← signvoice
                                                └─ Windows + Python 3.12
                                                     signvoice.py --server …   (webcam, microphone,
                                                                                speakers, robot screen)
```

| Part | Runs on | What it does |
|---|---|---|
| `ros/tiago_remote` | notebook, WSL Ubuntu 20.04 + ROS Noetic | web server: serves the phone page, sends the joystick to the robot (`/phone_vel`), relays the conversation between the phone and signvoice |
| `signvoice.py --server …` | notebook, Windows, Python 3.12 | the original signvoice app (camera, 250 ASL signs, LLM sentences, Whisper listen mode, voice, history) shown full screen on the robot's screen; with `--server` it also syncs with the phone |
| phone | any phone browser | **Drive** (joystick) and **Conversation** (remote buttons + history) |
| TIAGo | the robot | its own ROS (drivers, laser, navigation); receives velocity commands on `/phone_vel` |

Without `--server`, `signvoice.py` behaves exactly like the original version.

---

## 2. The robot

What we found on the lab's TIAGo (useful when something does not match the PAL documentation):

| Item | Value |
|---|---|
| Model | TIAGo **without arm** (PMB2 base), modified for the EnrichMe project (torso tablet, mini PC "AIS 27") |
| Robot computer | Ubuntu 14.04, **ROS Indigo**, hostname **`tiago-27c`** |
| Address on the cable | robot **`10.68.0.1`**; it gives the notebook an address by DHCP (e.g. `10.68.0.128`, domain `reem-lan`) |
| SSH | user `pal` (ask the lab for the password) |
| Head camera (Xtion) | **not working** — we use the notebook's webcam |
| Clock | stuck in 2009 (not synchronised); does not affect driving |
| Velocity input used | **`/phone_vel`**, a `twist_mux` input reserved for phones (priority 98, timeout 0.5 s) |

`twist_mux` priorities on the robot (`rosparam get /twist_mux/topics`):

| Input | Topic | Priority |
|---|---|---|
| joystick / tablet | `joy_vel` / `tab_vel` | 100 |
| marker | `marker_vel` | 99 |
| **phone (ours)** | **`phone_vel`** | **98** |
| rviz joystick | `rviz_joy_vel` | 95 |
| keyboard | `key_vel` | 90 |
| servoing | `servoing_cmd_vel` | 20 |
| navigation (move_base) | `nav_vel` | 10 |

Path to the wheels: `/phone_vel → /twist_mux → /speed_limit → /mobile_base_controller/cmd_vel`.
`speed_limit` is PAL's safety filter: it slows down or stops the robot when the laser sees an obstacle,
so the robot may refuse to move when it is close to a wall. Backwards speed is limited to 0.2 m/s by the robot.

---

## 3. Notebook setup (once)

Tested on Windows 11 with WSL 2.

### 3.1 WSL with Ubuntu 20.04

ROS Noetic only exists for **Ubuntu 20.04**. Ubuntu 22.04/24.04 do not work. Install 20.04 as an extra
distribution. ROS + Gazebo + the TIAGo simulation need about **15 GB**; if `C:` is short on space,
install it on another drive:

```powershell
wsl --update
mkdir D:\WSL
wsl --install -d Ubuntu-20.04 --location D:\WSL\Ubuntu-20.04
```

> If `C:` (or the drive holding the WSL disk) fills up during installation, the Linux disk gets corrupted
> (`Input/output error` everywhere). Free space, then `wsl --unregister Ubuntu-20.04` and reinstall.

### 3.2 ROS Noetic

In Ubuntu 20.04:

```bash
cd ~
sudo apt update && sudo apt upgrade -y
sudo apt install -y lsb-release curl gnupg
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt-get install -y git wget ipython3 python3-rosinstall ros-noetic-desktop-full \
  python3-catkin-tools python3-rosdep python-is-python3 ros-noetic-actionlib-tools ros-noetic-moveit-commander \
  python3-tornado unzip
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && source ~/.bashrc
rosversion -d          # -> noetic
```

### 3.3 TIAGo simulation workspace (only needed for simulation)

```bash
mkdir -p ~/tiago_public_ws && cd ~/tiago_public_ws
wget https://raw.githubusercontent.com/pal-robotics/tiago_tutorials/noetic-devel/tiago_public-noetic.rosinstall
rosinstall src /opt/ros/noetic tiago_public-noetic.rosinstall

# pal_msgs: since Oct 2025 the latest version no longer contains pal_navigation_msgs -> use 0.19.0
cd src/pal_msgs && git fetch --tags && git checkout 0.19.0 && cd ~/tiago_public_ws

sudo rosdep init
rosdep update --include-eol-distros          # Noetic is end-of-life: without this flag rosdep ignores it
rosdep install -y --from-paths src --ignore-src --rosdistro noetic --skip-keys "urdf_test omni_drive_controller orocos_kdl pal_filters libgazebo9-dev pal_usb_utils speed_limit_node camera_calibration_files pal_moveit_plugins pal_startup_msgs pal_local_joint_control pal_pcl_points_throttle_and_filter current_limit_controller hokuyo_node dynamixel_cpp pal_moveit_capabilities pal_pcl dynamic_footprint gravity_compensation_controller pal-orbbec-openni2 pal_loc_measure pal_map_manager ydlidar_ros_driver"

source /opt/ros/noetic/setup.bash
catkin build -DCATKIN_ENABLE_TESTING=0 -j $(expr `nproc` / 2)      # ~15-40 min
echo "source ~/tiago_public_ws/devel/setup.bash" >> ~/.bashrc
```

### 3.4 This package (`tiago_remote`)

Link the package from the repository into the workspace, so `git pull` is enough to update it
(adapt the path to where you cloned the repository):

```bash
ln -s "/mnt/d/<path-to>/ROB011/ros/tiago_remote" ~/tiago_public_ws/src/tiago_remote
cd ~/tiago_public_ws && catkin build tiago_remote && source devel/setup.bash
```

If you only want to drive the real robot (no simulation), a minimal workspace is enough:
`mkdir -p ~/robot_ws/src`, link the package into `~/robot_ws/src`, then `catkin build` there.

### 3.5 signvoice on Windows

signvoice needs **Python 3.12** (MediaPipe has no wheels for newer versions), on **Windows**
(the webcam is not reachable from WSL).

```powershell
winget install -e --id Python.Python.3.12          # then open a NEW PowerShell
cd <path-to>\ROB011
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # once, allows activating the venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Models path must not contain accents.** MediaPipe cannot open files in a path like
> `D:\Documentos - Não Apagar\…`. Keep the models in an ASCII path and pass `--models-dir`:
> `python signvoice.py --models-dir D:\signvoice_models` (copy the `models` folder there).

Optional LLM sentences (OpenRouter, free model), in the same PowerShell window before running:

```powershell
$env:LLM_API_KEY = "<your key>"                       # never commit the key
$env:LLM_MODEL = "nvidia/nemotron-3.5-super-vl:free"
```

The free plan has a daily request limit; when it runs out, sentences fall back to word-by-word.

### 3.6 WSL mirrored networking (needed for the real robot and the phone)

ROS 1 connects in both directions: the robot must be able to reach the ROS node inside WSL, and the phone
must reach the web server. With WSL's default (NAT) network this is impossible. Mirrored mode makes WSL use
the Windows network adapters directly.

```powershell
# normal PowerShell
Set-Content -Path "$env:USERPROFILE\.wslconfig" -Value "[wsl2]`nnetworkingMode=mirrored"
# PowerShell as administrator: allow incoming connections to WSL
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
wsl --shutdown
```

Check in Ubuntu (give it a few seconds after starting): `ip -4 addr` should list the same addresses as
`ipconfig` on Windows (the cable adapter `10.68.0.x` when the robot is plugged in, the Wi-Fi address, …).

To go back to normal: delete `%USERPROFILE%\.wslconfig` and run `wsl --shutdown`.

---

## 4. Try it in simulation

Drive the simulated TIAGo with the phone page and use the real translator on the notebook.

```bash
# terminal 1 (Ubuntu)
roslaunch tiago_2dnav_gazebo tiago_navigation.launch public_sim:=true lost:=false end_effector:=pal-gripper
# terminal 2 (Ubuntu), after Gazebo is up
source ~/tiago_public_ws/devel/setup.bash
roslaunch tiago_remote remote.launch
```

```powershell
# PowerShell, in the ROB011 folder with the venv active
python signvoice.py --server ws://127.0.0.1:8080/ws/translator --models-dir D:\signvoice_models
```

Open `http://localhost:8080`. The simulation runs at about 1/3 of real time in WSL, so the robot moves slowly.
Always start Gazebo **before** `remote.launch` (otherwise `remote.launch` starts its own ROS master and
Gazebo dies when you stop it).

---

## 5. Power the robot on and off

Controls are at the back of the base, above the orange band: red **emergency button**, small **main switch**
below it, display in the middle, round metal **On/Off button** on the right. The charger plugs in at the bottom.

**On**

1. Free space around the robot.
2. Release the emergency button (turn it so it pops up).
3. Main switch on.
4. Short press on the On/Off button (its light turns on).
5. Wait **~5 minutes**. The robot computer has no screen: the sign that it is up is the notebook getting
   a `10.68.0.x` address on the cable (see 6.1).

**Off**

1. Short press on the On/Off button (soft shutdown — do not hold it).
2. Wait 1–2 minutes until the light is off and the fans stop.
3. Main switch off.
   Only if nothing happens after 3 minutes: hold On/Off ~10 s to force it, then main switch off.

Do not use the emergency button to switch off (it only cuts the motors) and do not cut the main switch while
the computer is on. The charger can be unplugged while the robot is on; unplug it before driving.

---

## 6. Connect to the robot

### 6.1 Cable and address

Plug the robot's spare Ethernet port (yellow cable) into the notebook. The robot gives the notebook an address:

```powershell
ipconfig        # the Ethernet adapter should show 10.68.0.x, gateway 10.68.0.1, DNS suffix reem-lan
ping 10.68.0.1
```

If the adapter shows `169.254.x.x`, the robot computer is not up yet (wait) or the cable is in the wrong port.

### 6.2 SSH (optional, for checks)

```bash
ssh -o HostKeyAlgorithms=+ssh-rsa -o ServerAliveInterval=5 pal@10.68.0.1
```

- Answer `yes` the first time. The old robot needs `HostKeyAlgorithms=+ssh-rsa`. If it complains about
  key exchange or cipher, add `-o KexAlgorithms=+diffie-hellman-group1-sha1,diffie-hellman-group14-sha1 -o Ciphers=+aes128-cbc`.
- Avoid `-Y` (X11 forwarding): it froze the session. To leave a frozen session: `Enter`, `~`, `.`.

Useful checks inside the robot:

```bash
rosnode list | head
rostopic list | grep -E "phone_vel|scan|battery"
rostopic echo -n 1 /power/battery_level
rosparam get /twist_mux/topics
rostopic info /mobile_base_controller/cmd_vel      # publisher should be /speed_limit only
```

First motion test from inside the robot (see the [safety checklist](#9-safety-checklist) first) — slow left turn, stops by itself after 3 s:

```bash
timeout 3 rostopic pub -r 10 /phone_vel geometry_msgs/Twist '{angular: {z: 0.3}}'
```

### 6.3 Point ROS on the notebook to the robot

Requires mirrored networking (3.6). In Ubuntu, once:

```bash
echo "10.68.0.1 tiago-27c" | sudo tee -a /etc/hosts          # the robot announces itself by name
cat > ~/tiago_env.sh << 'EOF'
export ROS_MASTER_URI=http://10.68.0.1:11311
export ROS_IP=10.68.0.128
echo "ROS -> robot tiago-27c (10.68.0.1)"
EOF
```

`ROS_IP` must be the notebook's address on the cable (check `ipconfig`; it was `10.68.0.128` in our tests).
Run `source ~/tiago_env.sh` only in terminals that should talk to the real robot; other terminals keep
working with the simulation.

Check both directions:

```bash
source ~/tiago_env.sh
rostopic echo -n 1 /power/battery_level                                        # robot -> notebook
timeout 3 rostopic pub -r 10 /phone_vel geometry_msgs/Twist '{angular: {z: 0.3}}'   # notebook -> robot (it turns)
```

---

## 7. Run everything on the real robot

```bash
# Ubuntu — only ONE instance of the server
source ~/tiago_env.sh
source ~/tiago_public_ws/devel/setup.bash
roslaunch tiago_remote remote.launch
```

The output must show `ROS_MASTER_URI=http://10.68.0.1:11311` and must **not** show
`started core service [/rosout]` (that would mean a local ROS master was started instead of using the robot's).

```powershell
# PowerShell, ROB011 folder, venv active (+ LLM variables if wanted)
python signvoice.py --server ws://127.0.0.1:8080/ws/translator --models-dir D:\signvoice_models --fullscreen
```

Expected: `[remote] connected to ws://127.0.0.1:8080/ws/translator` in PowerShell,
`translator connected` in the Ubuntu terminal, and "Translator online" on the phone.

The signvoice window is the robot's screen: show it full screen on the monitor in the robot's torso.

### Phone

The phone must be on the same network as the notebook. School Wi-Fi blocks devices from seeing each other,
so use the **phone's hotspot**:

1. Turn on the hotspot on the phone and connect the notebook's Wi-Fi to it (the robot cable stays as is).
2. `ipconfig` → Wi-Fi IPv4 address (e.g. `172.20.10.2` on an iPhone hotspot).
3. On the phone: `http://172.20.10.2:8080` (http, not https).

If the page does not open, check that Ubuntu sees the new Wi-Fi address (`ip -4 addr | grep 172.20`);
if not, stop the server, `wsl --shutdown`, reopen Ubuntu and start the server again.

---

## 8. Using the phone remote

**Drive**

- Set **Max speed to 0.20 m/s** for first tests (up to 1.0 m/s).
- Push the joystick: up/down = forward/backward, left/right = turn. Release = the robot stops.
- The robot also stops if the phone disconnects, the page is minimised, or no command arrives for 0.5 s.
- **Stop** (bottom) stops the robot at any time.

**Conversation** (same actions as the buttons in the signvoice window)

| Control | Action |
|---|---|
| Talk / Listen | signs → voice / voice → text (Whisper; the speech model downloads on first use) |
| × on a sign | delete that sign |
| Undo / Clear | remove the last sign / all signs (in Listen: clear the caption) |
| Speak | speak the current sentence now (otherwise: lower your hands for 1.5 s) |
| History | "You" = signed sentences, "Them" = heard speech |
| LLM sentences | turn the LLM reconstruction on/off (grey if no `LLM_API_KEY`) |
| Robot full screen | full screen on/off for the signvoice window |

With `--server`, signvoice's pointing gestures are **off** (ASL signs such as YOU, ME, THERE also point and
could trigger buttons). Add `--gestures` to keep them. Keyboard and mouse still work on the notebook.

---

## 9. Safety checklist

Before any motion:

- [ ] Charger unplugged.
- [ ] At least 1 m of free space; nobody in the robot's path.
- [ ] Ethernet cable has slack (or the robot only turns in place).
- [ ] Someone has a hand on the **emergency button**.
- [ ] Max speed low (0.20 m/s) for the first tests.
- [ ] Only one `remote.launch` running.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Input/output error` everywhere in WSL | the disk holding WSL was full during installation | free space, `wsl --unregister Ubuntu-20.04`, reinstall (3.1) |
| `rosdep`: cannot locate `pal_navigation_msgs` | newer `pal_msgs` moved it out | `git checkout 0.19.0` in `src/pal_msgs` (3.3) |
| rosdep ignores Noetic | Noetic is end-of-life | `rosdep update --include-eol-distros` |
| `FileNotFoundError ... face_landmarker.task` | accent in the path (MediaPipe) | `--models-dir D:\signvoice_models` |
| `Activate.ps1 cannot be loaded` | PowerShell script policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `py` is not recognised | Python 3.12 not installed | `winget install -e --id Python.Python.3.12`, new PowerShell |
| `Invalid roslaunch XML syntax` | `--` inside an XML comment | fixed in the launch file; never write `--` in launch comments |
| `Address already in use` (port 8080) | another server instance is still running | `pkill -f remote_server.py`, start one instance; or `port:=8090` |
| `shutdown request: new node registered with same name` | two `remote.launch` started | keep only one; the old one now stops completely |
| `[remote] cannot reach ws://…` | server not running, or wrong address | start `remote.launch` first; use `127.0.0.1` rather than `localhost` |
| robot does not answer `ping` / notebook has `169.254.x.x` | robot computer still booting | wait ~5 min after power on |
| SSH session freezes | `-Y` X11 forwarding with the old robot | connect without `-Y`; leave with `Enter ~ .` |
| WSL: `Network is unreachable` right after `wsl --shutdown` | mirrored adapters not ready yet | wait a few seconds, check `ip -4 addr` |
| robot ignores the joystick near walls | PAL `speed_limit` safety filter | move to free space |
| backwards is slow | robot limit 0.2 m/s | expected |
| phone cannot open the page on school Wi-Fi | client isolation | use the phone hotspot (7) |

---

## 11. Reference

### Repository layout

```
ROB011/
├── signvoice.py            translator (original app + optional --server / --gestures)
├── requirements.txt        Python 3.12 dependencies (Windows)
├── README.md               signvoice user guide
├── docs/TIAGO_GUIDE.md     this guide
└── ros/tiago_remote/       ROS Noetic package (WSL)
    ├── scripts/remote_server.py   web server + ROS bridge
    ├── web/index.html             phone page (Drive / Conversation)
    └── launch/remote.launch       args: port, cmd_vel_topic, max_linear, max_angular
```

### WebSocket messages (`/ws/translator`)

| Direction | Message |
|---|---|
| signvoice → server | `{"type": "state", "mode": "talk"\|"listen", "tokens": [...], "sentence", "heard", "listen_status", "history": [["you"\|"them", text], ...], "llm", "llm_available", "fullscreen", "speaking"}` (on change, at least every 2 s) |
| server → signvoice | `{"type": "action", "action": "mode"\|"speak"\|"undo"\|"clear"\|"del"\|"llm"\|"fullscreen", "value"?, "arg"?}` |

### ROS interface (`tiago_remote`)

| Topic | Type | Notes |
|---|---|---|
| `/phone_vel` (out) | geometry_msgs/Twist | joystick; published only while the joystick is in use, so autonomous navigation is not blocked |
| `/conversation/history` (out) | std_msgs/String, latched | JSON list of `["you"\|"them", text]` |

### Quick start (after setup)

```bash
source ~/tiago_env.sh && source ~/tiago_public_ws/devel/setup.bash && roslaunch tiago_remote remote.launch
```
```powershell
python signvoice.py --server ws://127.0.0.1:8080/ws/translator --models-dir D:\signvoice_models --fullscreen
```
Phone: `http://<notebook Wi-Fi IP>:8080`.
