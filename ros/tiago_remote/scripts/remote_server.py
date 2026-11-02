#!/usr/bin/env python3
"""
tiago_remote — web server that bridges the phone UI to ROS.

Phone  --(HTTP + WebSocket /ws)-->  this node  --(ROS)-->  TIAGo (simulation or real)
Robot screen  <--(WebSocket /ws/screen)--  big captions page  (http://<host>:<port>/screen.html)
signvoice_robot.py  <--(WebSocket /ws/recognizer)-->  webcam frames, face box, signs and sentence

Tabs of the UI and what they do here
  DRIVE      joystick                      -> <cmd_vel_topic> (default /phone_vel, a twist_mux input)
  TRANSLATE  live image + head + captions  -> /head_controller/point_head_action, /head_controller/command
                                           <- /sign/caption, /sign/glosses
             signs are recognised only while a face is in view (/translation_active)

ROS interface
  publishes  <cmd_vel_topic>        geometry_msgs/Twist
  publishes  /robot_mode            std_msgs/String  (latched)  drive | translate
  publishes  /translation_active    std_msgs/Bool    (latched)  translate mode AND person in view
  publishes  /head_controller/command  trajectory_msgs/JointTrajectory
  action     /head_controller/point_head_action  control_msgs/PointHead
  subscribes /person/face           std_msgs/String  JSON {"visible", "u", "v", "w", "h"} (normalised 0..1)
  subscribes /sign/caption, /sign/glosses  std_msgs/String
  subscribes /joint_states, <camera_topic> (CompressedImage), <camera_info_topic>

Safety
  * velocities clamped to ~max_linear / ~max_angular
  * robot stops if no joystick command for ~cmd_timeout s, or when the phone disconnects
  * motion is only accepted in DRIVE mode
"""
import json
import math
import os
import signal
import time

import rospy
import rospkg
import actionlib
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState, CameraInfo, CompressedImage
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import PointHeadAction, PointHeadGoal

import tornado.ioloop
import tornado.web
import tornado.websocket

MODES = ("drive", "translate")
ZERO_BURST = 3            # zero-velocity messages sent after a stop
FACE_TIMEOUT = 1.0        # s without a detection -> "no one in view"
TARGET = (0.5, 0.33)      # where the face should sit in the image (upper third: leaves room for the hands)
CENTRE_TOL = (0.08, 0.08)
STEP_DEG = 6.0            # head step for the arrow buttons
FIND_STEPS = 4            # max iterations of "find face automatically"


def clamp(v, limit):
    return max(-limit, min(limit, float(v)))


class Bridge(object):
    def __init__(self):
        self.max_lin = float(rospy.get_param("~max_linear", 1.0))
        self.max_ang = float(rospy.get_param("~max_angular", 1.5))
        self.timeout = float(rospy.get_param("~cmd_timeout", 0.5))
        self.rate = float(rospy.get_param("~publish_rate", 10.0))
        self.cam_fps = float(rospy.get_param("~camera_fps", 10.0))
        self.cam_frame = rospy.get_param("~camera_frame", "xtion_rgb_optical_frame")
        self.cam_topic = rospy.get_param("~camera_topic", "/xtion/rgb/image_raw/compressed")
        cmd_topic = rospy.get_param("~cmd_vel_topic", "/phone_vel")
        info_topic = rospy.get_param("~camera_info_topic", "/xtion/rgb/camera_info")
        # True: camera image from ROS (simulation / Xtion). False: images come from signvoice_robot (webcam).
        self.use_ros_camera = bool(rospy.get_param("~use_ros_camera", False))

        self.cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)
        self.mode_pub = rospy.Publisher("/robot_mode", String, queue_size=1, latch=True)
        self.active_pub = rospy.Publisher("/translation_active", Bool, queue_size=1, latch=True)
        self.head_pub = rospy.Publisher("/head_controller/command", JointTrajectory, queue_size=1)
        self.point_client = actionlib.SimpleActionClient("/head_controller/point_head_action", PointHeadAction)

        rospy.Subscriber("/sign/caption", String, self.on_caption)
        rospy.Subscriber("/sign/glosses", String, self.on_glosses)
        rospy.Subscriber("/person/face", String, self.on_face)
        rospy.Subscriber("/joint_states", JointState, self.on_joints)
        if self.use_ros_camera:
            rospy.Subscriber(info_topic, CameraInfo, self.on_info)
        self.img_sub = None

        self.loop = None
        self.clients = set()      # phones
        self.screens = set()      # robot screen(s): receive status + captions only
        self.recognizer = None    # signvoice_robot connection
        self.rec_cfg = None
        self.mode = "drive"
        self.lin = self.ang = 0.0
        self.last_cmd = 0.0
        self.zero_left = 0
        self.caption, self.glosses = "", []
        self.face, self.face_stamp = None, 0.0
        self.pan = self.tilt = 0.0
        self.K = None                     # (fx, fy, cx, cy, W, H)
        self.find_left, self.find_next = 0, 0.0
        self.last_frame = 0.0
        self.active = None

        self.mode_pub.publish(String(data=self.mode))
        self.set_active(False)
        rospy.loginfo("tiago_remote: velocity -> %s (max %.2f m/s, %.2f rad/s)", cmd_topic, self.max_lin, self.max_ang)

    # ------------------------------------------------------------------ state
    def visible(self):
        return self.face is not None and (time.time() - self.face_stamp) < FACE_TIMEOUT

    def centred(self):
        return self.visible() and abs(self.face["u"] - TARGET[0]) < CENTRE_TOL[0] \
            and abs(self.face["v"] - TARGET[1]) < CENTRE_TOL[1]

    def status(self):
        return {"type": "status", "mode": self.mode,
                "recognizer": self.recognizer is not None,
                "visible": self.visible(), "centred": self.centred(),
                "face": self.face if self.visible() else None, "target": TARGET,
                "pan": round(math.degrees(self.pan)), "tilt": round(math.degrees(self.tilt)),
                "max_linear": self.max_lin, "max_angular": self.max_ang}

    def set_active(self, value):
        if value != self.active:
            self.active = value
            self.active_pub.publish(Bool(data=value))

    # ----------------------------------------------------------------- motion
    def publish_twist(self, lin, ang):
        t = Twist()
        t.linear.x, t.angular.z = lin, ang
        self.cmd_pub.publish(t)

    def stop(self):
        self.lin = self.ang = 0.0
        self.zero_left = ZERO_BURST
        self.publish_twist(0.0, 0.0)

    # ------------------------------------------------------------------- head
    def head_to(self, pan, tilt, secs=1.0):
        traj = JointTrajectory()
        traj.joint_names = ["head_1_joint", "head_2_joint"]
        p = JointTrajectoryPoint()
        p.positions = [pan, tilt]
        p.time_from_start = rospy.Duration(secs)
        traj.points = [p]
        self.head_pub.publish(traj)

    def point_at_px(self, u, v):
        """Turn the head so that the camera's optical axis goes through pixel (u, v)."""
        if self.K is None:
            return "No camera_info yet"
        if not self.point_client.wait_for_server(rospy.Duration(0.05)):
            return "Head action server not available"
        fx, fy, cx, cy, _, _ = self.K
        target = PointStamped()
        target.header.frame_id = self.cam_frame
        target.header.stamp = rospy.Time(0)
        target.point.x, target.point.y, target.point.z = (u - cx) / fx, (v - cy) / fy, 1.0
        goal = PointHeadGoal()
        goal.target = target
        goal.pointing_frame = self.cam_frame
        goal.pointing_axis.z = 1.0
        goal.min_duration = rospy.Duration(0.4)
        goal.max_velocity = 0.8
        self.point_client.send_goal(goal)
        return None

    def head_step(self, direction):
        if direction == "center":
            self.head_to(0.0, 0.0)
            return None
        if self.K is None:
            return "No camera_info yet"
        fx, fy, cx, cy, _, _ = self.K
        d = math.tan(math.radians(STEP_DEG))
        du = {"left": -1, "right": 1}.get(direction, 0) * fx * d
        dv = {"up": -1, "down": 1}.get(direction, 0) * fy * d
        return self.point_at_px(cx + du, cy + dv)

    def find_step(self):
        """One iteration of 'find face': aim so the face lands on TARGET."""
        if not self.visible() or self.centred():
            self.find_left = 0
            return
        fx, fy, cx, cy, W, H = self.K
        fu, fv = self.face["u"] * W, self.face["v"] * H
        self.point_at_px(fu - (TARGET[0] * W - cx), fv - (TARGET[1] * H - cy))
        self.find_left -= 1

    # ------------------------------------------------------------ periodic tick
    def tick(self):
        now = time.time()
        # motion
        fresh = (now - self.last_cmd) < self.timeout
        moving = self.lin != 0.0 or self.ang != 0.0
        if self.mode == "drive" and fresh and moving:
            self.publish_twist(self.lin, self.ang)
            self.zero_left = ZERO_BURST
        else:
            if moving and not fresh:
                rospy.logwarn_throttle(2.0, "tiago_remote: joystick timeout, stopping")
                self.lin = self.ang = 0.0
            if self.zero_left > 0:
                self.publish_twist(0.0, 0.0)
                self.zero_left -= 1
        # auto face finding
        if self.find_left > 0 and now >= self.find_next:
            self.find_next = now + 0.9
            self.find_step()
        # translation runs only with someone in view
        self.set_active(self.mode == "translate" and self.visible())
        # camera stream only when someone looks at the CAMERA tab
        want_stream = self.mode == "translate" and bool(self.clients)
        if self.use_ros_camera:
            self.update_camera_sub(want_stream)
        self.update_recognizer(want_stream)
        if self.clients or self.screens:
            self.broadcast(self.status(), targets=self.clients | self.screens)

    def update_recognizer(self, stream):
        """Tell signvoice_robot whether to stream frames and whether to recognise signs."""
        cfg = {"type": "config", "stream": stream, "active": bool(self.active)}
        if self.recognizer is not None and cfg != self.rec_cfg:
            self.rec_cfg = cfg
            self.recognizer.write_message(json.dumps(cfg))

    def update_camera_sub(self, want):
        if want and self.img_sub is None:
            self.img_sub = rospy.Subscriber(self.cam_topic, CompressedImage, self.on_image, queue_size=1, buff_size=2 ** 22)
        elif not want and self.img_sub is not None:
            self.img_sub.unregister()
            self.img_sub = None

    # --------------------------------------------------------------- messages
    def handle(self, msg):
        kind = msg.get("type")
        if kind == "cmd":
            if self.mode == "drive":
                self.lin = clamp(msg.get("lin", 0.0), self.max_lin)
                self.ang = clamp(msg.get("ang", 0.0), self.max_ang)
                self.last_cmd = time.time()
        elif kind == "stop":
            rospy.loginfo("tiago_remote: STOP pressed")
            self.find_left = 0
            self.stop()
        elif kind == "mode":
            return self.set_mode(msg.get("mode"))
        elif kind == "head":
            self.find_left = 0
            err = self.head_step(msg.get("dir"))
            if err:
                return toast(err)
        elif kind == "find_face":
            if self.K is None:
                return toast("No camera_info yet")
            if not self.visible():
                return toast("No face in view")
            self.find_left, self.find_next = FIND_STEPS, 0.0
        elif kind == "ping":
            return {"type": "pong", "t": msg.get("t")}
        return None

    def set_mode(self, mode):
        if mode not in MODES or mode == self.mode:
            return None
        self.mode = mode
        self.find_left = 0
        self.stop()
        self.mode_pub.publish(String(data=mode))
        rospy.loginfo("tiago_remote: mode -> %s", mode)
        return None

    def broadcast(self, payload, binary=False, targets=None):
        data = payload if binary else json.dumps(payload)
        for c in list(self.clients if targets is None else targets):
            try:
                c.write_message(data, binary=binary)
            except tornado.websocket.WebSocketClosedError:
                self.clients.discard(c)
                self.screens.discard(c)

    # ------------------------------------------------ signvoice_robot messages
    def on_recognizer(self, msg):
        kind = msg.get("type")
        if kind == "hello":
            # webcam intrinsics from its horizontal field of view (no calibration file needed)
            W, H = float(msg.get("width", 640)), float(msg.get("height", 480))
            fx = (W / 2.0) / math.tan(math.radians(float(msg.get("hfov", 60.0))) / 2.0)
            self.K = (fx, fx, W / 2.0, H / 2.0, W, H)
            rospy.loginfo("tiago_remote: recognizer connected (%dx%d, hfov %.0f deg)", W, H, msg.get("hfov", 60.0))
        elif kind == "face":
            if msg.get("visible"):
                self.face = {k: float(msg.get(k, 0.0)) for k in ("u", "v", "w", "h")}
                self.face_stamp = time.time()
        elif kind == "sign":
            self.caption, self.glosses = msg.get("caption", ""), list(msg.get("glosses", []))
            self.broadcast({"type": "caption", "caption": self.caption, "glosses": self.glosses},
                           targets=self.clients | self.screens)

    def on_recognizer_frame(self, jpeg):
        if self.mode == "translate":
            self.broadcast(jpeg, True)

    # -------------------------------------------------- ROS callbacks (threads)
    def on_caption(self, msg):
        self.caption = msg.data
        self._push_caption()

    def on_glosses(self, msg):
        self.glosses = msg.data.split()
        self._push_caption()

    def _push_caption(self):
        if self.loop is not None:
            self.loop.add_callback(self.broadcast, {"type": "caption", "caption": self.caption, "glosses": self.glosses},
                                   False, self.clients | self.screens)

    def on_face(self, msg):
        try:
            f = json.loads(msg.data)
        except ValueError:
            return
        if f.get("visible"):
            self.face = {k: float(f.get(k, 0.0)) for k in ("u", "v", "w", "h")}
            self.face_stamp = time.time()

    def on_joints(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name == "head_1_joint":
                self.pan = pos
            elif name == "head_2_joint":
                self.tilt = pos

    def on_info(self, msg):
        if msg.K[0] > 0:
            self.K = (msg.K[0], msg.K[4], msg.K[2], msg.K[5], float(msg.width), float(msg.height))

    def on_image(self, msg):
        now = time.time()
        if now - self.last_frame < 1.0 / self.cam_fps or self.loop is None:
            return
        self.last_frame = now
        self.loop.add_callback(self.broadcast, bytes(msg.data), True)


def toast(text):
    return {"type": "toast", "text": text}


class WSHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, bridge):
        self.bridge = bridge

    def check_origin(self, origin):
        return True

    def open(self):
        self.bridge.clients.add(self)
        rospy.loginfo("tiago_remote: phone connected (%d client(s))", len(self.bridge.clients))
        self.write_message(json.dumps(self.bridge.status()))
        self.write_message(json.dumps({"type": "caption", "caption": self.bridge.caption, "glosses": self.bridge.glosses}))

    def on_message(self, message):
        try:
            reply = self.bridge.handle(json.loads(message))
        except (ValueError, TypeError) as e:
            rospy.logwarn("tiago_remote: bad message %r (%s)", message, e)
            return
        if reply:
            self.write_message(json.dumps(reply))

    def on_close(self):
        self.bridge.clients.discard(self)
        rospy.loginfo("tiago_remote: phone disconnected (%d left)", len(self.bridge.clients))
        if not self.bridge.clients:
            self.bridge.stop()


class ScreenHandler(tornado.websocket.WebSocketHandler):
    """Robot screen: read-only (status + captions)."""
    def initialize(self, bridge):
        self.bridge = bridge

    def check_origin(self, origin):
        return True

    def open(self):
        self.bridge.screens.add(self)
        rospy.loginfo("tiago_remote: robot screen connected")
        self.write_message(json.dumps(self.bridge.status()))
        self.write_message(json.dumps({"type": "caption", "caption": self.bridge.caption, "glosses": self.bridge.glosses}))

    def on_close(self):
        self.bridge.screens.discard(self)


class RecognizerHandler(tornado.websocket.WebSocketHandler):
    """signvoice_robot.py: sends webcam frames (binary JPEG), face boxes and signs; receives config."""
    def initialize(self, bridge):
        self.bridge = bridge

    def check_origin(self, origin):
        return True

    def open(self):
        self.bridge.recognizer = self
        self.bridge.rec_cfg = None

    def on_message(self, message):
        if isinstance(message, bytes):
            self.bridge.on_recognizer_frame(message)
            return
        try:
            self.bridge.on_recognizer(json.loads(message))
        except (ValueError, TypeError) as e:
            rospy.logwarn("tiago_remote: bad recognizer message (%s)", e)

    def on_close(self):
        if self.bridge.recognizer is self:
            self.bridge.recognizer = None
            rospy.logwarn("tiago_remote: recognizer disconnected")


def main():
    rospy.init_node("tiago_remote", disable_signals=True)
    bridge = Bridge()

    web_dir = os.path.join(rospkg.RosPack().get_path("tiago_remote"), "web")
    port = int(rospy.get_param("~port", 8080))
    app = tornado.web.Application([
        (r"/ws", WSHandler, {"bridge": bridge}),
        (r"/ws/screen", ScreenHandler, {"bridge": bridge}),
        (r"/ws/recognizer", RecognizerHandler, {"bridge": bridge}),
        (r"/(.*)", tornado.web.StaticFileHandler, {"path": web_dir, "default_filename": "index.html"}),
    ])
    app.listen(port, address="0.0.0.0")

    loop = tornado.ioloop.IOLoop.current()
    bridge.loop = loop
    tornado.ioloop.PeriodicCallback(bridge.tick, 1000.0 / bridge.rate).start()

    def shutdown(*_):
        bridge.stop()
        loop.add_callback_from_signal(loop.stop)
        rospy.signal_shutdown("ctrl-c")

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    rospy.loginfo("tiago_remote: phone  -> http://localhost:%d/  (or http://<PC-IP>:%d/)", port, port)
    rospy.loginfo("tiago_remote: screen -> http://localhost:%d/screen.html", port)
    loop.start()


if __name__ == "__main__":
    main()
