# TIAGo Navigation Remote

This server is kept outside the `ros` folder because it runs on the Windows
PC and connects to ROS remotely through rosbridge.

This is the minimal phone navigation service for Windows. It does not expose
conversation, history, translator or LLM features.

The browser sends joystick commands to the local HTTP server. The server
forwards bounded `geometry_msgs/Twist` messages to the robot through the same
rosbridge connection used by `signvoice.py`.

Start it together with signvoice:

```powershell
python signvoice.py --ros-camera --nav-server
```

Open `http://<PC-IP>:8080` on the phone. Hold a direction button to move;
releasing it or pressing **STOP** sends zero velocity. The single vertical
slider controls `linear.x` (forward/backward), proportionally to how far the
knob is pushed, with a maximum of `0.25 m/s`. The **TURN LEFT**
and **TURN RIGHT** buttons control `angular.z` while held; releasing a button
stops rotation. There is no lateral translation. The top of the slider
commands forward and the bottom commands backward. Commands stop automatically
after 0.5 seconds without a refresh.

Use **NAVIGATION: OFF** to disable the manual topic. The server then
unadvertises `/phone_vel`, allowing the lower-priority face-tracking topic to
control the rotation again. **NAVIGATION: ON** advertises the manual topic
again and restores manual-navigation priority.

Manual navigation uses `/phone_vel`. With the current TIAGo mux, automatic
face-centering should use `/key_vel` because that topic is already accepted by
the mux. Start `signvoice.py` with `--base-command-topic /key_vel`. The mux
gives `/phone_vel` priority `98` and `/key_vel` priority `90`, so manual
navigation overrides automatic centering when both are active.

Options:

```text
--nav-port PORT
--nav-command-topic TOPIC
```

The default ROS topic is `/phone_vel`.

When started by `signvoice.py` on Windows, this navigation process runs at
below-normal priority. The camera capture and sign-processing process remain
the priority, so joystick HTTP requests are not allowed to take precedence
over frame processing. If the preview still loses FPS, reduce the processing
load with `--no-face` or increase `--infer-every`; the camera itself remains
independent from the navigation server.
