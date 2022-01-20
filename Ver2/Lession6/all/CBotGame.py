from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

from gtts import gTTS
from playsound import playsound

from audio import list_audio_devices
from audio import AudioInput
from soundfile import SoundFile

import pyttsx3 as pt
import speech_recognition as sr
from google.cloud import speech
import os
import io

import threading

from handDetector import handDetector

GSTREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "cbotvoice-330303-6a7f0d4b95d9.json"

# Creates google client
client = speech.SpeechClient()

# voice input-----
r = sr.Recognizer()

# voice output-----
engine = pt.init()

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open("face.pickle", "rb").read())

scale_percent = 60

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
writer = None
time.sleep(2.0)

def recordVoice(record_time):

    
    # create the audio device
    input_mic = AudioInput(mic=11, sample_rate=24000, chunk_size=4096)

    # loop until user exits
    sample_count_end = 24000*record_time 
    sample_count = 0

    # setup exit signal handler        
    record = True

    # create the output wav
    output_wav = SoundFile('test.wav', mode='w', samplerate=24000, channels=1)

    while record:
        print("Say something pls ... ")
        samples = input_mic.next()
        output_wav.write(samples)
        sample_count += len(samples)
        if sample_count > sample_count_end:
            record=False

        output_wav.close()

    filename = 'test.wav'

    with sr.AudioFile(filename) as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        audio = speech.RecognitionAudio(content=audio_data.get_wav_data(convert_rate=24000, convert_width=2))

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        audio_channel_count=1,
        language_code="vi-VN",
        sample_rate_hertz=24000,
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    response = operation.result(timeout=90)
    # Sends the request to google to transcribe the audio
    # response = client.recognize(request={"config": config, "audio": audio})

    # Reads the response
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print(u"Transcript: {}".format(result.alternatives[0].transcript))
        print("Confidence: {}".format(result.alternatives[0].confidence))

    return result.alternatives[0].transcript

def speak(sent):
    tts = gTTS(sent, tld='com.vn', lang='vi')
    tts.save('voice.mp3')
    playsound('voice.mp3')

def face_reg(frame, model):

    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    rgb = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    r = frame.shape[1] / float(rgb.shape[1])
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb,
                                            model=model)
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)

    return names

def getNumber(ar):
    s = ""
    for i in ar:
        s += str(ar[i]);

    if (s == "00000"):
        return (0)
    elif (s == "01000"):
        return (1)
    elif (s == "01100"):
        return (2)
    elif (s == "01110"):
        return (3)
    elif (s == "01111"):
        return (4)
    elif (s == "11111"):
        return (5)
    elif (s == "01001"):
        return (6)
    elif (s == "01011"):
        return (7)

def GuessNumber():
    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    writer = None
    time.sleep(2.0)

    pTime = 0
    detector = handDetector(detectionCon=0.75)
    while True:
        success, img = cap.read()
        img = detector.findHands(img, draw=True)
        lmList = detector.findPosition(img, draw=False)
        # print(lmList)
        tipId = [4, 8, 12, 16, 20]
        if (len(lmList) != 0):
            fingers = []
            # thumb
            if (lmList[tipId[0]][1] > lmList[tipId[0] - 1][1]):
                fingers.append(1)
            else:
                fingers.append(0)
            # 4 fingers
            for id in range(1, len(tipId)):

                if (lmList[tipId[id]][2] < lmList[tipId[id] - 2][2]):
                    fingers.append(1)

                else:
                    fingers.append(0)

            cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(getNumber(fingers)), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 20)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 3)
        cv2.imshow("image", img)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

def GuessColor():

    boundaries = [
	    ([17, 15, 100], [50, 56, 200]), # Xanh
	    ([86, 31, 4], [220, 88, 50]),  # Vang
	    ([25, 146, 190], [62, 174, 250]), # Do
	    ([103, 86, 65], [145, 133, 128]) # Trang
    ]

    # Video Capturing class from OpenCV		
    video_capture = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        cv2.namedWindow("Color Detection Window", cv2.WINDOW_AUTOSIZE)
	    count = 0
        while True:
            return_key, imageFrame = video_capture.read()
            imageFrame = imageFrame[50:430, 100:540]
	        h, w, _ = imageFrame.shape

            if not return_key:
                break
	        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
            # Set range for green color and  
            # define mask 
             
            area = []

            for (color_lower, color_upper) in boundaries:
                color_mask = cv2.inRange(hsvFrame, color_lower, color_upper) 
        
                # For each color 
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

                area.apped(max_area_color)
            
        max_id = 0

        for index in range(0, len(area)):
            if area[index] > area[max_id]:
                max_id = index

        if (max_id == 0):
            speak("Xanh")
        elif (max_id == 1):
            speak("Vang")
        elif (max_id == 2):
            speak("Do")
        elif (max_id == 3):
            speak("Trang")

        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print("Cannot open Camera")

count = 0
count_ = 0

masterName = "Van"
notMaster = True

while notMaster:
    ret, frame = vs.read()

    count_ = count_ + 1
    # print(count_)

    if count_ > 100:

        names = face_reg(frame, model="hog")

        for name in names:
                if name == masterName:
                    count = count + 1
                    if count > 10:
                        print("Hello master")
                        speak("Xin chào bạn tôi rất ngu và không chạy được")
                        notMaster = False

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

cv2.destroyAllWindows()

print("Check: " + str(count_))

speak("Xin chào")
speak("Mình là C bot")
speak("Bạn có khỏe không")
time.sleep(1.0)
speak("Game không bạn")

game = "None"

answer = recordVoice(5)

if answer == "Đoán số":
    print("Bạn ngu vl")
    game = "number"
if answer == "Đoán màu":
    print("Bạn vẫn ngu vl")
    game = "color"

time.sleep(2.0)

print("[INFO] starting video stream...")
vs.release()

if game == "Number":
    GuessNumber()
elif game == "color":
    GuessColor()
else :
    speak("Cút")


