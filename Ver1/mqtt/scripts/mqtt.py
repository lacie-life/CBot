#!/usr/bin/env python2

import rospy
from std_msgs.msg import String

import time
import sys
import paho.mqtt.client as mqtt
import paho.mqtt.subscribe as subscribe

pub = rospy.Publisher('/voice_control/voice_control', String, queue_size=10)
rospy.init_node('mqtt')
rate = rospy.Rate(10) # 10hz

def cmd(msg):
    pub.publish(String(msg))

# The callback for when the client receives a CONNACK response from the server.
#khoi tao ket noi server MQTT brocker
def on_connect(client, userdata, flags, rc):
    #print("Connected with result code "+str(rc))
    #thay doi topic tren server, moi robot 1 topic server khac nhau
    client.subscribe("garden1/sensor1")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
#    print(msg.topic+" "+str(msg.payload))
    msg_ = msg.payload.decode('utf-8')
    print(type(msg_))
    cmd(msg_)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("citlab.myftp.org", 1883, 60)

client.loop_forever()

