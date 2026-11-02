#!/usr/bin/env python3
"""
Publishes a demo sentence on /sign/glosses and /sign/caption while /translation_active is true
(TRANSLATE mode with someone in view), to test the UI before the real recogniser exists.
"""
import rospy
from std_msgs.msg import String, Bool

DEMO = [
    ("HELLO", "Hello"),
    ("MY", "Hello, my"),
    ("NAME", "Hello, my name"),
    ("A-N-A", "Hello, my name is Ana"),
    ("OLD", "Hello, my name is Ana,"),
    ("22", "Hello, my name is Ana, I'm 22 years old."),
]


def main():
    rospy.init_node("fake_signs")
    gloss_pub = rospy.Publisher("/sign/glosses", String, queue_size=1)
    cap_pub = rospy.Publisher("/sign/caption", String, queue_size=1)
    state = {"active": False, "since": rospy.get_time()}

    def on_active(msg):
        state["active"] = msg.data
        state["since"] = rospy.get_time()

    rospy.Subscriber("/translation_active", Bool, on_active)

    rate = rospy.Rate(1.0 / 1.1)
    i, glosses = 0, []
    while not rospy.is_shutdown():
        if state["active"]:
            if i >= len(DEMO):
                i, glosses = 0, []
            glosses.append(DEMO[i][0])
            gloss_pub.publish(String(data=" ".join(glosses)))
            cap_pub.publish(String(data=DEMO[i][1]))
            i += 1
        elif rospy.get_time() - state["since"] > 3.0 and glosses:
            # inactive for a while: start a new sentence next time
            i, glosses = 0, []
            gloss_pub.publish(String(data=""))
            cap_pub.publish(String(data=""))
        rate.sleep()


if __name__ == "__main__":
    main()
