#!/usr/bin/env python
import rospy
import time
from std_msgs.msg import String

def control_motor(msg):
	pub.publish(msg)

def ColorDetect(msg):
	pub1.publish("1")
	print("run")

def control_servo(msg):
	pub2.publish(msg)
	
def Color(msg):
	print("2")
	control_servo("open")
	control_motor("forward")
	time.sleep(2)
	control_motor("stop")
	control_servo("close")


# initialization
if __name__ == '__main__':
	rospy.init_node('main')
	pub1 = rospy.Publisher('/camera/camera', String, queue_size=10)	
	pub = rospy.Publisher('/jetbot_motors/cmd_str', String, queue_size=10)
	pub2 = rospy.Publisher('/servo/control_servo', String, queue_size=10)
	rospy.Subscriber('camera/detected', String, Color)
	rospy.Subscriber('camera/cmd', String, ColorDetect)
	rate = rospy.Rate(10)
	rospy.spin()


