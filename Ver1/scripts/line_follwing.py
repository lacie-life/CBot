#!/usr/bin/env python

import rospy
import time
from std_msgs.msg import String

import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(7,GPIO.IN)
GPIO.setup(13,GPIO.IN)
GPIO.setup(15,GPIO.IN)

def cmd(msg):
    pub = rospy.Publisher('/jetbot_motors/cmd_str', String, queue_size=10)
    rospy.init_node('line_following')
    rate = rospy.Rate(10) # 10hz
    pub.publish(String(msg))

if __name__ == '__main__':
    try:
	while(True):
		x=GPIO.input(7)
	     	y=GPIO.input(13)
		z=GPIO.input(15)
	 
		if x==0 and y==0 and z==1:
		   print("Re trai")
		   cmd("left")
		if x==1 and y==0 and z==1:
		   print("Di thang")
		   cmd("forward")
		if x==1 and y==0 and z==0:
		   print("Re phai")
		   cmd("right")
		if x==0 and y==0 and z==0:
		   print("Dung lai")
		   cmd("stop")
		if x==1 and y==1 and z==1:
		   print("Dung lai")
		   cmd("stop")
		time.sleep(0.1)
        
    except rospy.ROSInterruptException:
        pass




