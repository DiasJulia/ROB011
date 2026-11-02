#!/usr/bin/env python3
"""
Simulated face detector (Gazebo only).

Instead of looking at pixels, it takes the true position of the "person" model in Gazebo,
projects its head into the robot's camera and publishes the result exactly like a real
detector would. This lets the CAMERA / TRANSLATE flow be tested with a simple dummy person.

publishes /person/face  std_msgs/String  JSON {"visible", "u", "v", "w", "h"}  (normalised 0..1)
"""
import json

import numpy as np
import rospy
import tf2_ros
from gazebo_msgs.msg import ModelStates, LinkStates
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from tf.transformations import quaternion_matrix


def pose_to_mat(position, orientation):
    m = quaternion_matrix([orientation.x, orientation.y, orientation.z, orientation.w])
    m[:3, 3] = [position.x, position.y, position.z]
    return m


def project(T_world_cam, p_world, K, width, height, face_radius=0.11):
    """Project a 3D world point into the image. Returns the face dict (normalised)."""
    p_cam = np.linalg.inv(T_world_cam).dot(np.append(p_world, 1.0))[:3]
    x, y, z = p_cam
    if z < 0.3 or z > 8.0:
        return {"visible": False}
    fx, fy, cx, cy = K
    u, v = fx * x / z + cx, fy * y / z + cy
    w = 2.0 * fx * face_radius / z
    h = 1.25 * w
    if not (0 <= u <= width and 0 <= v <= height):
        return {"visible": False}
    return {"visible": True, "u": u / width, "v": v / height, "w": w / width, "h": h / height, "dist": float(z)}


class SimFaceDetector(object):
    def __init__(self):
        self.person = rospy.get_param("~person_model", "person")
        self.face_height = float(rospy.get_param("~face_height", 1.62))
        self.head_link = rospy.get_param("~head_link", "head_2_link")
        self.cam_frame = rospy.get_param("~camera_frame", "xtion_rgb_optical_frame")
        self.models = self.links = self.info = None
        self.T_head_cam = None
        self.tf_buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buf)
        self.pub = rospy.Publisher("/person/face", String, queue_size=1)
        rospy.Subscriber("/gazebo/model_states", ModelStates, lambda m: setattr(self, "models", m), queue_size=1)
        rospy.Subscriber("/gazebo/link_states", LinkStates, lambda m: setattr(self, "links", m), queue_size=1)
        rospy.Subscriber(rospy.get_param("~camera_info_topic", "/xtion/rgb/camera_info"), CameraInfo,
                         lambda m: setattr(self, "info", m), queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / float(rospy.get_param("~rate", 15.0))), self.update)
        rospy.loginfo("sim_face_detector: watching Gazebo model '%s'", self.person)

    def head_to_cam(self):
        if self.T_head_cam is None:
            t = self.tf_buf.lookup_transform(self.head_link, self.cam_frame, rospy.Time(0), rospy.Duration(0.2))
            self.T_head_cam = pose_to_mat(t.transform.translation, t.transform.rotation)
        return self.T_head_cam

    def update(self, _event):
        if self.models is None or self.links is None or self.info is None:
            return
        try:
            i_person = self.models.name.index(self.person)
            i_head = next(i for i, n in enumerate(self.links.name) if n.endswith("::" + self.head_link))
            T_head_cam = self.head_to_cam()
        except (ValueError, StopIteration):
            rospy.logwarn_throttle(5.0, "sim_face_detector: person model or robot head not found in Gazebo")
            return
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return
        lp = self.links.pose[i_head]
        T_world_cam = pose_to_mat(lp.position, lp.orientation).dot(T_head_cam)
        pp = self.models.pose[i_person].position
        face = np.array([pp.x, pp.y, pp.z + self.face_height])
        K = (self.info.K[0], self.info.K[4], self.info.K[2], self.info.K[5])
        self.pub.publish(String(data=json.dumps(project(T_world_cam, face, K, self.info.width, self.info.height))))


if __name__ == "__main__":
    rospy.init_node("sim_face_detector")
    SimFaceDetector()
    rospy.spin()
