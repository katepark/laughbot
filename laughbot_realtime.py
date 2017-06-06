#The Cloud Speech API v1 is officially released and is generally available from the https://speech.googleapis.com/v1/speech endpoint.
#Should probably move the service_account_key file (CS224s-Laughbot-cdb7a14ba039.json) to local directories instead of shared for data safety and permissions reasons, after you login
'''Run this:
	export GOOGLE_APPLICATION_CREDENTIALS=./service_account_key.json #set environment variable to key path
	Follow instructions for Google cloud SDK installation: https://cloud.google.com/sdk/downloads 
	https://cloud.google.com/sdk/docs/quickstarts
	(to change compute zone: gcloud config set compute/zone NAME)
	"Created a default .boto configuration file at [/Users/nataliemuenster/.boto]. See this file and
		[https://cloud.google.com/storage/docs/gsutil/commands/config] for more
		information about configuring Google Cloud Storage.
		Your Google Cloud SDK is configured and ready to use!
		"

	gcloud auth application-default login (just hit y when it asks and it will redirect you to authenticate)
	gcloud auth activate-service-account --key-file=./service_account_key.json

	Using speech API: (https://cloud.google.com/speech/docs/getting-started)
	gcloud auth application-default print-access-token:
	#ex result: ya29.ElpfBCz3lei0FaA_8OJ88fssq5qArUkdQjdqVhSNKGajTk-6J38JSKCiIWeTArj4e9eZhgHhEOcO0SS0qWkm9n21KcOnagEWWEEWy3MaH5r0wHJxqfSEfXK4kC4
	call in console:
	"curl -s -k -H "Content-Type: application/json" \
    -H "Authorization: Bearer access_token" \
    https://speech.googleapis.com/v1/speech:recognize \
    -d @sync-request.json"

    #You can also batch long audio files for speech recognition (using Asynchronous Recognition requests) or have the Speech API listen to a stream and return results while the stream is open using a gRPC StreamingRecognize request. For example, a streaming recognition task may provide transcription from a user while they are speaking. --> make it realtime with package? https://github.com/Uberi/speech_recognition/blob/master/examples/background_listening.py

	#https://pypi.python.org/pypi/SpeechRecognition/
    pip install SpeechRecognition
    Google API Client Library for Python (required only if you need to use the Google Cloud Speech API, recognizer_instance.recognize_google_cloud) pip install google-api-python-client
    brew install portaudio && sudo brew link portaudio
    pip install pyaudio
'''


import numpy as np
import speech_recognition as sr
import pyaudio
import wave

audioFile = "latest_recording.wav"

#http://sharewebegin.blogspot.com/2013/07/record-from-mic-python.html
def record_audio():
	CHUNK = 1024 
	FORMAT = pyaudio.paInt16 #paInt8 #pyaudio.paFloat32?
	CHANNELS = 2 #1?
	RATE = 44100 #sample rate
	RECORD_SECONDS = 7
	WAVE_OUTPUT_FILENAME = "output.wav"

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
	                channels=CHANNELS,
	                rate=RATE,
	                input=True,
	                frames_per_buffer=CHUNK) #buffer
	#should be  output=True, not input?

	print("* recording")

	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	    data = stream.read(CHUNK)
	    frames.append(data) # 2 bytes(16 bits) per channel

	print("* done recording")

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(audioFile, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()


def get_transcript_from_file():
	#open file "latest_recording.wav"
	# use the audio file as the audio source
	r = sr.Recognizer()
	with sr.AudioFile("./" + audioFile) as source:
	    audio = r.record(source)  # read the entire audio file

	# recognize speech using Google Cloud Speech
	#GOOGLE_CLOUD_SPEECH_CREDENTIALS = #If default credentials not working from the environment variable, insert credentials here and add to r.recognize_google_cloud call
	try:
		line = r.recognize_google_cloud(audio)
		print("Google Cloud Speech thinks you said " + line)
		return line
	except sr.UnknownValueError:
		print("Google Cloud Speech could not understand audio")
	except sr.RequestError as e:
		print("Could not request results from Google Cloud Speech service; {0}".format(e))


def get_transcript_from_mic():
	# obtain audio from the microphone
	r = sr.Recognizer()
	with sr.Microphone() as source:
	    print("Say something and laughbot will decide whether you're funny!")
	    audio = r.listen(source)

	# recognize speech using Google Cloud Speech
	#If credentials not specified, the library will try to automatically `find the default API credentials JSON file <https://developers.google.com/identity/protocols/application-default-credentials>`__.
	#currently this default is an environment variable set as "./service_account_key.json". used to be CS224s-Laughbot-cdb7a14ba039.json
	#GOOGLE_CLOUD_SPEECH_CREDENTIALS = {} #If default credentials not working from the environment variable, insert credentials here and add to r.recognize_google_cloud call
	try:
		line = r.recognize_google_cloud(audio)
		print("Google Cloud Speech thinks you said: " + line)#, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS))
		return line
	except sr.UnknownValueError:
		print("Google Cloud Speech could not understand audio")
	except sr.RequestError as e:
		print("Could not request results from Google Cloud Speech service; {0}".format(e))




if __name__ == "__main__":
	#use multithreading to get keyboard press to stop? much more elegant than ctrl-c..self.
	x = 1
	while(x==1): #will be while true
		#multithread to get transcription AND audio recording for mfcc extraction
		#speech = get_transcript_from_mic()
		record_audio()
		print "audio recorded"
		transcript = get_transcript_from_file()
		#process speech with audioFile and transcript
		print "transcript", transcript
		x = 0







