#!/usr/bin/env python3
"""
tiago_remote — web server that bridges the phone UI, the robot and the signvoice translator.

    Phone (browser)  --WebSocket /ws-->            this node  --ROS-->  TIAGo (simulation or real)
    signvoice.py --server ...  --WebSocket /ws/translator-->  (conversation state <-> phone buttons)

Phone tabs
  DRIVE         joystick -> <cmd_vel_topic> (default /phone_vel, a twist_mux input)
  CONVERSATION  Talk/Listen, signs, sentence, history, Undo/Clear/Speak/x, LLM, full screen
                (the translator runs signvoice.py on the robot's notebook, with its own screen)

ROS interface
  publishes  <cmd_vel_topic>          geometry_msgs/Twist
  publishes  /conversation/history    std_msgs/String (latched)  JSON [["you"|"them", text], ...]

Safety
  * velocities clamped to ~max_linear / ~max_angular
  * the robot stops if no joystick command arrives for ~cmd_timeout s, or when the phone disconnects
"""
import json
import os
import signal
import time

import rospy
import rospkg
from geometry_msgs.msg import Twist
from std_msgs.msg import String

import tornado.ioloop
import tornado.web
import tornado.websocket

ZERO_BURST = 3            # zero-velocity messages sent after a stop
ACTIONS = ("mode", "speak", "undo", "clear", "del", "llm", "fullscreen")


def clamp(v, limit):
    return max(-limit, min(limit, float(v)))


class Bridge(object):
    def __init__(self):
        self.max_lin = float(rospy.get_param("~max_linear", 1.0))
        self.max_ang = float(rospy.get_param("~max_angular", 1.5))
        self.timeout = float(rospy.get_param("~cmd_timeout", 0.5))
        self.rate = float(rospy.get_param("~publish_rate", 10.0))
        cmd_topic = rospy.get_param("~cmd_vel_topic", "/phone_vel")

        self.cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)
        self.hist_pub = rospy.Publisher("/conversation/history", String, queue_size=1, latch=True)

        self.clients = set()       # phones
        self.translator = None     # signvoice.py --server
        self.conv = None           # last conversation state from the translator
        self.history_json = ""
        self.lin = self.ang = 0.0
        self.last_cmd = 0.0
        self.zero_left = 0
        rospy.loginfo("tiago_remote: velocity -> %s (max %.2f m/s, %.2f rad/s)", cmd_topic, self.max_lin, self.max_ang)

    # ----------------------------------------------------------------- motion
    def publish_twist(self, lin, ang):
        t = Twist()
        t.linear.x, t.angular.z = lin, ang
        self.cmd_pub.publish(t)

    def stop(self):
        self.lin = self.ang = 0.0
        self.zero_left = ZERO_BURST
        self.publish_twist(0.0, 0.0)

    def tick(self):
        """Publishes only while the joystick is in use, so autonomous navigation is not blocked."""
        fresh = (time.time() - self.last_cmd) < self.timeout
        moving = self.lin != 0.0 or self.ang != 0.0
        if fresh and moving:
            self.publish_twist(self.lin, self.ang)
            self.zero_left = ZERO_BURST
        else:
            if moving and not fresh:
                rospy.logwarn_throttle(2.0, "tiago_remote: joystick timeout, stopping")
                self.lin = self.ang = 0.0
            if self.zero_left > 0:
                self.publish_twist(0.0, 0.0)
                self.zero_left -= 1

    # --------------------------------------------------------- phone messages
    def handle(self, msg):
        kind = msg.get("type")
        if kind == "cmd":
            self.lin = clamp(msg.get("lin", 0.0), self.max_lin)
            self.ang = clamp(msg.get("ang", 0.0), self.max_ang)
            self.last_cmd = time.time()
        elif kind == "stop":
            rospy.loginfo("tiago_remote: STOP pressed")
            self.stop()
        elif kind == "action":
            if msg.get("action") not in ACTIONS:
                return None
            if self.translator is None:
                return {"type": "toast", "text": "Translator offline"}
            self.translator.write_message(json.dumps(msg))
        elif kind == "ping":
            return {"type": "pong", "t": msg.get("t")}
        return None

    def broadcast(self, payload):
        data = json.dumps(payload)
        for c in list(self.clients):
            try:
                c.write_message(data)
            except tornado.websocket.WebSocketClosedError:
                self.clients.discard(c)

    def conv_payload(self):
        return {"type": "conv", "online": self.translator is not None, "state": self.conv}

    # ---------------------------------------------------- translator messages
    def on_translator(self, msg):
        if msg.get("type") != "state":
            return
        self.conv = msg
        hist = json.dumps(msg.get("history", []))
        if hist != self.history_json:
            self.history_json = hist
            self.hist_pub.publish(String(data=hist))
        self.broadcast(self.conv_payload())


class PhoneHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, bridge):
        self.bridge = bridge

    def check_origin(self, origin):
        return True

    def open(self):
        self.bridge.clients.add(self)
        rospy.loginfo("tiago_remote: phone connected (%d client(s))", len(self.bridge.clients))
        self.write_message(json.dumps(self.bridge.conv_payload()))

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


class TranslatorHandler(tornado.websocket.WebSocketHandler):
    """signvoice.py --server ws://<host>:<port>/ws/translator"""
    def initialize(self, bridge):
        self.bridge = bridge

    def check_origin(self, origin):
        return True

    def open(self):
        self.bridge.translator = self
        rospy.loginfo("tiago_remote: translator connected")
        self.bridge.broadcast(self.bridge.conv_payload())

    def on_message(self, message):
        try:
            self.bridge.on_translator(json.loads(message))
        except (ValueError, TypeError) as e:
            rospy.logwarn("tiago_remote: bad translator message (%s)", e)

    def on_close(self):
        if self.bridge.translator is self:
            self.bridge.translator = None
            rospy.logwarn("tiago_remote: translator disconnected")
            self.bridge.broadcast(self.bridge.conv_payload())


def main():
    rospy.init_node("tiago_remote", disable_signals=True)
    bridge = Bridge()

    web_dir = os.path.join(rospkg.RosPack().get_path("tiago_remote"), "web")
    port = int(rospy.get_param("~port", 8080))
    app = tornado.web.Application([
        (r"/ws", PhoneHandler, {"bridge": bridge}),
        (r"/ws/translator", TranslatorHandler, {"bridge": bridge}),
        (r"/(.*)", tornado.web.StaticFileHandler, {"path": web_dir, "default_filename": "index.html"}),
    ])
    # all interfaces, IPv4 and IPv6 ("localhost" on Windows may resolve to ::1 first)
    app.listen(port)

    loop = tornado.ioloop.IOLoop.current()
    tornado.ioloop.PeriodicCallback(bridge.tick, 1000.0 / bridge.rate).start()

    def shutdown(*_):
        bridge.stop()
        loop.add_callback_from_signal(loop.stop)
        rospy.signal_shutdown("ctrl-c")

    # if ROS shuts this node down (e.g. another node with the same name started),
    # stop the web server too instead of leaving a page that can no longer move the robot
    rospy.on_shutdown(lambda: loop.add_callback(loop.stop))

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    rospy.loginfo("tiago_remote: phone      -> http://localhost:%d/  (or http://<PC-IP>:%d/)", port, port)
    rospy.loginfo("tiago_remote: translator -> python signvoice.py --server ws://127.0.0.1:%d/ws/translator", port)
    loop.start()


if __name__ == "__main__":
    main()
