#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from std_msgs.msg import String
import time
from Adafruit_MotorHAT import Adafruit_MotorHAT
#from std_msgs.msg import String
#from simple_pid import PID


GSTREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=1/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'



def ColorDetect(msg):
    if(msg.data == "1"):
	color_lower = np.array([22, 93, 0], np.uint8) 
        color_upper = np.array([45, 255, 255], np.uint8)

    # Video Capturing class from OpenCV		
    video_capture = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        cv2.namedWindow("Color Detection Window", cv2.WINDOW_AUTOSIZE)
	count = 0;
        while True:
            return_key, imageFrame = video_capture.read()
            imageFrame = imageFrame[50:430, 100:540]
	    h, w, _ = imageFrame.shape
	#    print('width: ', w)
        #    print('height:', h)
            if not return_key:
                break
	    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
            # Set range for green color and  
            # define mask 
             
            color_mask = cv2.inRange(hsvFrame, color_lower, color_upper) 
        

            # For green color 
	    kernal = np.ones((5, 5), "uint8")
            color_mask = cv2.dilate(color_mask, kernal) 
            res_color = cv2.bitwise_and(imageFrame, imageFrame, 
                                mask = color_mask) 
 

            # Creating contour to track green color 
            contours, hierarchy = cv2.findContours(color_mask, 
                                                cv2.RETR_TREE, 
                                                cv2.CHAIN_APPROX_SIMPLE) 

            max_area_color = 0
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > max_area_color:
                    max_area_color = area
            
            for pic, contour in enumerate(contours): 
                area = cv2.contourArea(contour) 
                if(area == max_area_color and area > 800): 
                    x, y, w, h = cv2.boundingRect(contour) 
                    imageFrame = cv2.rectangle(imageFrame, (x, y),  
                                            (x + w, y + h), 
                                            (0, 255, 0), 2) 
                    tam_x = x+w/2
                    tam_y = y+h/2
                    data = str(tam_x) +" "+str(tam_y)
		 
		    if ((int(tam_x) > 200) and (int(tam_x) < 240)):
			on_cmd_str("stop")
			time.sleep(0.05)
        		print("stop")
			count=count+1
			
   		    elif (int(tam_x) >= 240 and int(tam_x) <= 440):
       			on_cmd_str("right")
			count = 0
			time.sleep(0.05)
			on_cmd_str("stop")
			print("right")
    		    elif (int(tam_x) <= 200 and int(tam_x) >= 0):
       			on_cmd_str("left")
			count = 0
			time.sleep(0.05)
			on_cmd_str("stop")
			print("left")
			
		    print(data)
		    
                    #talker(data) 
                    cv2.putText(imageFrame, "color Colour", (x, y), 
                                cv2.FONT_HERSHEY_SIMPLEX,  
                                1.0, (0, 255, 0)) 
        

            cv2.imshow("Color Detection Window", imageFrame)
	    time.sleep(0.2)
	    print ("Stop counting: ",count)

            key = cv2.waitKey(30) & 0xff
            # Stop the program on the ESC key
            if key == 27:
                break
	    if count == 5:
		pub.publish("1")
		break

        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print("Cannot open Camera")
   # Color("yes")
def set_speed(motor_ID, value):
	max_pwm = 210.0
	speed = int(min(max(abs(value * max_pwm), 0), max_pwm))
	a = b = 0

	if motor_ID == 1:
		motor = motor_left
		a = 1
		b = 0
	elif motor_ID == 2:
		motor = motor_right
		a = 2
		b = 3
	else:
		rospy.logerror('set_speed(%d, %f) -> invalid motor_ID=%d', motor_ID, value, motor_ID)
		return
	
	motor.setSpeed(speed)
	if value < 0:
		motor.run(Adafruit_MotorHAT.FORWARD)
		motor.MC._pwm.setPWM(a,0,0)
		motor.MC._pwm.setPWM(b,0,speed*16)
	elif value > 0:
		motor.run(Adafruit_MotorHAT.BACKWARD)
		motor.MC._pwm.setPWM(a,0,speed*16)
		motor.MC._pwm.setPWM(b,0,0)
	else:
		motor.run(Adafruit_MotorHAT.RELEASE)
		motor.MC._pwm.setPWM(a,0,0)
		motor.MC._pwm.setPWM(b,0,0)

def on_cmd_str(msg):

	if msg == "left":
		set_speed(motor_left_ID,  -0.85)
		set_speed(motor_right_ID,  0.85) 
	elif msg == "right":
		set_speed(motor_left_ID,   0.85)
		set_speed(motor_right_ID, -0.85) 
	elif msg == "forward":
		set_speed(motor_left_ID,   0.7)
		set_speed(motor_right_ID,  0.7)
	elif msg == "backward":
		set_speed(motor_left_ID,  -0.7)
		set_speed(motor_right_ID, -0.7)  
	elif msg == "stop":
		all_stop()

# stops all motors
def all_stop():
	set_speed(motor_left_ID, 0.0)
	set_speed(motor_right_ID, 0.0)
#def Color(msg):
#    	pub1 = rospy.Publisher('/main/Camera', String, queue_size=10)
#	rospy.init_node('camera')
#	rate = rospy.Rate(10)
#	pub1.publish(msg)
if __name__ == "__main__":
# setup motor controller
	motor_driver = Adafruit_MotorHAT(i2c_bus=1)

	motor_left_ID = 1
	motor_right_ID = 2

	motor_left = motor_driver.getMotor(motor_left_ID)
	motor_right = motor_driver.getMotor(motor_right_ID)

	# stop the motors as precaution
	all_stop()
	rospy.init_node('camera')
	rospy.Subscriber('camera/camera', String, ColorDetect)
	pub = rospy.Publisher('/camera/detected', String, queue_size=10)
	#ColorDetect("1")
	rospy.spin()
	all_stop()
    
