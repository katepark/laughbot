# Portions of this page are modifications based on work created and shared by Google and used according to terms described in the Creative Commons 3.0 Attribution License.
import numpy as np
import os
# uncomment this line to suppress Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import speech_recognition as sr
import pyaudio
import wave
import threading
from threading import Thread
import subprocess

from convertaudiosample import *
from languagemodel import predictLaughter
from rnn import Config, RNNModel
from rnn_utils import *
from util import readExamples

audioFile = "laughbot_audio.wav" #"laughtrack8.wav"
transcriptFile = "laughbot_text.txt"

# based on code from
# http://sharewebegin.blogspot.com/2013/07/record-from-mic-python.html
def record_audio():
	exitKey = []
	Thread(target=end_recording, args=(exitKey,)).start()

	CHUNK = 1024 
	FORMAT = pyaudio.paInt16 #paInt8
	CHANNELS = 2 
	RATE = 44100 #sample rate
	RECORD_SECONDS = 60 #max time for audio input -- press enter to end earlier
	WAVE_OUTPUT_FILENAME = "output.wav"

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
	                channels=CHANNELS,
	                rate=RATE,
	                input=True,
	                frames_per_buffer=CHUNK) #buffer

	print("* recording")

	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		if exitKey:
			break
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


def get_transcript_from_file(credential):
	#open file "latest_recording.wav"
	# use the audio file as the audio source
	r = sr.Recognizer()
	with sr.AudioFile("./" + audioFile) as source:
	    audio = r.record(source)  # read the entire audio file

	# recognize speech using Google Cloud Speech
	try:
		line = r.recognize_google_cloud(audio, credentials_json = credential)
		# line = r.recognize_sphinx(audio) # workaround w/ sphinx for now
		line = line.replace("hahaha", "[Laughter]")
		line = line.replace("Ha-Ha", "[Laughter]")

		file = open(transcriptFile, 'w')
		file.write("0 " + line)
		file.close()

		return line
	except sr.UnknownValueError:
		print("Sorry, didn't get that.") #Google Cloud Speech could not understand audio
	except sr.RequestError as e:
		print("There seems to be a problem with transcribing your speech. Error with Google Cloud Speech service; {0}".format(e))

	return "no audio"

#this is for realtime, but doesn't save audio file
'''def get_transcript_from_mic():
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
'''

def end_recording(exitKey):
	raw_input()
	print "key pressed!"
	exitKey.append(None)
	

def play_laughtrack():
	print "LOL!"
	laughFiles = ["laughtracks/laughtrack{}.wav".format(i) for i in range(1, 8)]
	rand = np.random.randint(0,len(laughFiles))
	return_code = subprocess.call(["afplay", laughFiles[rand]])

if __name__ == "__main__":
	print("\n")
	print("Hi! I'm Laughbot! Talk to me and press the Enter key when you want me to decide whether you're funny.")
	print("--------------------------------------------------------------------------")

	# set up google cloud credential
	with open ('./service_account_key.json', 'r') as f:
		credential = f.read()

	with tf.Graph().as_default():
		model = RNNModel()
		init = tf.global_variables_initializer()

		with tf.Session() as session:
			session.run(init)
			# Load pretrained model
			print("Loading in model")
			new_saver = tf.train.import_meta_graph('saved_models/model.meta', clear_devices=True)
			new_saver.restore(session, 'saved_models/model')

		    # main REPL loop
			response = raw_input("Press 's' to start: ")
			while response != 'q':
			    # print("press enter to stop recording")
			    # record_audio()
			    # print("audio recorded")
			    transcript = get_transcript_from_file(credential)
			    print("transcript: ", transcript)
			    # convert_audio_sample()
			    
			    test_dataset = load_dataset("laughbot_audio.test.pkl")
			    feature_b, label_b, seqlens_b = make_batches(test_dataset, batch_size=len(test_dataset[0]))
			    feature_b = pad_all_batches(feature_b)
			    batch_cost, summary, acc, predicted, acoustic = model.train_on_batch(session, feature_b[0], label_b[0], seqlens_b[0], train=False)
			    text = readExamples('laughbot_text.txt')
			    prediction = predictLaughter(text, acoustic)
			    if prediction[0] == 1:
			        play_laughtrack()
			    else:
			    	print('Not funny :(')
			    response = raw_input("Press 'c' to continue, 'q' to quit: ")

			print('Thanks for talking to me')





