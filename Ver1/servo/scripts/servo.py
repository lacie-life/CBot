#!/usr/bin/env python3

# SDA = pin.SDA_1
# SCL = pin.SCL_1
# SDA_1 = pin.SDA
# SCL_1 = pin.SCL

from adafruit_servokit import ServoKit
import board
import busio
import time
import rospy
from std_msgs.msg import String

#from approxeng.input.selectbinder import ControllerResource


# On the Jetson Nano
# Bus 0 (pins 28,27) is board SCL_1, SDA_1 in the jetson board definition file
# Bus 1 (pins 5, 3) is board SCL, SDA in the jetson definition file
# Default is to Bus 1; We are using Bus 0, so we need to construct the busio first ...
#print("Initializing Servos")
i2c_bus0=(busio.I2C(board.SCL_1, board.SDA_1))
#print("Initializing ServoKit")
kit = ServoKit(channels=16, i2c=i2c_bus0)
# kit[0] is the bottom servo
# kit[1] is the top servo
#print("Done initializing")
# initialization
print("oke")
# simple string commands (left/right/forward/backward/stop)
def servo_control(msg):
	rospy.loginfo(rospy.get_caller_id() + '%s', msg.data)
	print(msg.data)
	if msg.data.lower() == "close":
		kit.servo[0].angle= 0
	elif msg.data.lower() == "open":
		kit.servo[0].angle= 180

if __name__ == '__main__':

# setup ros node
   rospy.init_node('servo')	
   rospy.Subscriber('~control_servo', String, servo_control)
# start running
   rospy.spin()
