#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from std_msgs.msg import String

GSTREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'

def faceDetect():

    # Video Capturing class from OpenCV		
    video_capture = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        cv2.namedWindow("Face Detection Window", cv2.WINDOW_AUTOSIZE)

        while True:
            return_key, imageFrame = video_capture.read()
           # imageFrame = imageFrame[50:430, 100:540]
            if not return_key:
                break

            hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
            red_lower = np.array([136, 87, 111], np.uint8)
            red_upper = np.array([180, 255, 255], np.uint8) 
            red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 

            # Set range for green color and  
            # define mask 
            green_lower = np.array([0, 220, 0], np.uint8) 
            green_upper = np.array([50, 255, 50], np.uint8) 
            green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 
        
            # Set range for blue color and 
            # define mask 
            blue_lower = np.array([100,100,100], np.uint8) 
            blue_upper = np.array([120, 255, 255], np.uint8) 
            blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 

            kernal = np.ones((5, 5), "uint8") 
            red_mask = cv2.dilate(red_mask, kernal) 
            res_red = cv2.bitwise_and(imageFrame, imageFrame,  
                              mask = red_mask)

            # For green color 
            green_mask = cv2.dilate(green_mask, kernal) 
            res_green = cv2.bitwise_and(imageFrame, imageFrame, 
                                mask = green_mask) 
      
            # For blue color 
            blue_mask = cv2.dilate(blue_mask, kernal) 
            res_blue = cv2.bitwise_and(imageFrame, imageFrame, 
                                    mask = blue_mask) 
   
            # Creating contour to track red color 
            contours, hierarchy = cv2.findContours(red_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 

            max_area_red = 0
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > max_area_red:
                    max_area_red = area
            
            for pic, contour in enumerate(contours): 
                area = cv2.contourArea(contour)
            #    print(contour)
                if(area == max_area_red and area > 800): 
                    x, y, w, h = cv2.boundingRect(contour) 
                    imageFrame = cv2.rectangle(imageFrame, (x, y),  
                                       (x + w, y + h),  
                                       (0, 0, 255), 2) 
                       
              
                    cv2.putText(imageFrame, "Red Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                        (0, 0, 255)) 

            # Creating contour to track green color 
            contours, hierarchy = cv2.findContours(green_mask, 
                                                cv2.RETR_TREE, 
                                                cv2.CHAIN_APPROX_SIMPLE) 

            max_area_green = 0
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > max_area_green:
                    max_area_green = area
            
            for pic, contour in enumerate(contours): 
                area = cv2.contourArea(contour) 
                if(area == max_area_green and area > 800): 
                    x, y, w, h = cv2.boundingRect(contour) 
                    imageFrame = cv2.rectangle(imageFrame, (x, y),  
                                            (x + w, y + h), 
                                            (0, 255, 0), 2) 
                    tam_x = x+w/2
                    tam_y = y+h/2
                    data = str(tam_x) +" "+str(tam_y)
		    print(data)
                    talker(data) 
                    cv2.putText(imageFrame, "Yellow Colour", (x, y), 
                                cv2.FONT_HERSHEY_SIMPLEX,  
                                1.0, (0, 255, 0)) 
        
            # Creating contour to track blue color 
            contours, hierarchy = cv2.findContours(blue_mask, 
                                                cv2.RETR_TREE, 
                                                cv2.CHAIN_APPROX_SIMPLE) 

            max_area_blue = 0
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > max_area_blue:
                    max_area_blue = area

            for pic, contour in enumerate(contours): 
                area = cv2.contourArea(contour) 
                if(area == max_area_blue and area > 800): 
                    x, y, w, h = cv2.boundingRect(contour) 
                    imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                            (x + w, y + h), 
                                            (255, 0, 0), 2) 
                   
                    cv2.putText(imageFrame, "Blue Colour", (x, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1.0, (255, 0, 0)) 

            cv2.imshow("Face Detection Window", imageFrame)

            key = cv2.waitKey(30) & 0xff
            # Stop the program on the ESC key
            if key == 27:
                break

        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print("Cannot open Camera")
def talker(msg):
    pub = rospy.Publisher('/control_motor_color/data', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    pub.publish(String(msg))

if __name__ == "__main__":
    faceDetect()
    
