import numpy as np
import pickle
import subprocess

MAX_AUDIO_VECTOR = 50

# TODO: call extract.sh here
subprocess.call("./extract.sh", shell=True)

outFile = open('laughbot_audio.test.pkl', 'wb')
mfccs = np.genfromtxt('laughbot_audio.csv', delimiter=',', skip_header=1)
endMfcc = len(mfccs)
startMfcc = max(0, endMfcc - MAX_AUDIO_VECTOR)

audio = mfccs[startMfcc:endMfcc, 2:]
pickle.dump(([audio], [0], [len(audio)]), outFile)