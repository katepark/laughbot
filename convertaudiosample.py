import numpy as np
import pickle
import subprocess

MAX_AUDIO_VECTOR = 50
MAX_BATCHES = 10


def convert_audio_sample():
	subprocess.call("./extract.sh", shell=True)

	outFile = open('laughbot_audio.test.pkl', 'wb')
	mfccs = np.genfromtxt('laughbot_audio.csv', delimiter=',', skip_header=1)
	# grab last 50
	endMfcc = len(mfccs)
	startMfcc = max(0, endMfcc - MAX_AUDIO_VECTOR)
	audio = mfccs[startMfcc:endMfcc, 2:]

	# use average of all
	count = 1
	endMfcc -= MAX_AUDIO_VECTOR
	startMfcc = max(0, endMfcc - MAX_AUDIO_VECTOR)
	while startMfcc > 0 and count < MAX_BATCHES:
		audio += mfccs[startMfcc:endMfcc, 2:]
		count += 1
		endMfcc -= MAX_AUDIO_VECTOR
		startMfcc = max(0, endMfcc - MAX_AUDIO_VECTOR)

	audio /= count
	pickle.dump(([audio], [0], [len(audio)]), outFile)