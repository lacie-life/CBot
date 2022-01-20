#!/usr/bin/env python3
# coding: utf-8

import pyttsx3 as pt
import speech_recognition as sr
from google.cloud import speech
import os
import io
from gtts import gTTS
from playsound import playsound
from audio import list_audio_devices
from audio import AudioInput
from soundfile import SoundFile

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "cbotvoice-ec175f40ed8d.json"

# Creates google client
client = speech.SpeechClient()

# voice input-----
r = sr.Recognizer()

# voice output-----
engine = pt.init()

# list_audio_devices()

# create the audio device
input_mic = AudioInput(mic=11, sample_rate=24000, chunk_size=4096)

def speak(sent):
    tts = gTTS(sent, tld='com.vn', lang='vi')
    tts.save('voice.mp3')
    playsound('voice.mp3')

# with sr.Microphone() as source:
#     print("Hey Buddy Say Something :")
#     a = r.listen(source)
#     audio = speech.RecognitionAudio(content=a.get_wav_data(convert_rate=24000, convert_width=2))

# loop until user exits
sample_count_end = 24000*5 # record 10s
sample_count = 0

# setup exit signal handler        
record = True

# create the output wav
output_wav = SoundFile('test.wav', mode='w', samplerate=24000, channels=1)

while record:
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

    if result.alternatives[0].transcript == "Xin chào":
        speak("Xin chào bạn tôi rất ngu và không chạy được")
    elif result.alternatives[0].transcript == "bạn là ai":
        speak("Tôi là một chú giun heo màu vàng nâu")




