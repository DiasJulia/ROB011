#!/usr/bin/env python3
"""
Real face detector (OpenCV Haar cascade) for the real robot.

subscribes <camera_topic>  sensor_msgs/CompressedImage  (default /xtion/rgb/image_raw/compressed)
publishes  /person/face    std_msgs/String JSON {"visible", "u", "v", "w", "h"}  (normalised 0..1)

Will be replaced later by the MediaPipe pipeline used for sign recognition
(same output topic, so the rest of the system does not change).
"""
import json
import os
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

CASCADE = "haarcascade_frontalface_default.xml"
SEARCH = [getattr(getattr(cv2, "data", None), "haarcascades", ""),
          "/usr/share/opencv4/haarcascades/", "/usr/share/opencv/haarcascades/",
          "/usr/local/share/opencv4/haarcascades/"]


def find_cascade():
    for d in SEARCH:
        p = os.path.join(d, CASCADE)
        if d and os.path.isfile(p):
            return p
    raise RuntimeError("Haar cascade not found. Try: sudo apt install opencv-data")


class FaceDetector(object):
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(find_cascade())
        self.period = 1.0 / float(rospy.get_param("~rate", 8.0))
        self.last = 0.0
        self.pub = rospy.Publisher("/person/face", String, queue_size=1)
        topic = rospy.get_param("~camera_topic", "/xtion/rgb/image_raw/compressed")
        rospy.Subscriber(topic, CompressedImage, self.on_image, queue_size=1, buff_size=2 ** 22)
        rospy.loginfo("face_detector: OpenCV Haar on %s", topic)

    def on_image(self, msg):
        now = time.time()
        if now - self.last < self.period:
            return
        self.last = now
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        H, W = img.shape
        faces = self.cascade.detectMultiScale(cv2.equalizeHist(img), scaleFactor=1.15, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            out = {"visible": False}
        else:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])   # biggest face = closest person
            out = {"visible": True, "u": (x + w / 2.0) / W, "v": (y + h / 2.0) / H, "w": w / float(W), "h": h / float(H)}
        self.pub.publish(String(data=json.dumps(out)))


if __name__ == "__main__":
    rospy.init_node("face_detector")
    FaceDetector()
    rospy.spin()
