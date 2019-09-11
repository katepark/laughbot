from flask import render_template
from app import app
import os, sys
import numpy as np
import tensorflow as tf
# ignore warnings
tf.logging.set_verbosity(tf.logging.ERROR)
import speech_recognition as sr
import pyaudio
import wave
import threading
from threading import Thread
import subprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# add parent directory to filepath for imports
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from rnn import RNNModel
from rnn_utils import *
from util import readExamples
from convertaudiosample import *
from languagemodel import fitModel

# audio params
CHUNK = 1024
FORMAT = pyaudio.paInt16 # paInt8
CHANNELS = 2 
RATE = 44100 # sample rate
WAVE_OUTPUT_FILENAME = "output.wav"
frames = []
p = None
stream = None
audioFile = "laughbot_audio.wav"
audioFileProcessed = "laughbot_audio.test.pkl"
savedlogmodelfile = 'logistic_model.sav'
savedvocabularyfile = 'vocabulary.sav'
savedfreqcolidxfile = 'freqcolidx.sav'

def callback(in_data, frame_count, time_info, status):
	frames.append(in_data)
	return (in_data, pyaudio.paContinue)

def get_transcript_from_file(credential):
	r = sr.Recognizer()
	with sr.AudioFile("./" + audioFile) as source:
	    audio = r.record(source)  # read the entire audio file

	# recognize speech using Google Cloud Speech
	try:
		text = r.recognize_google_cloud(audio, credentials_json = credential)
		text = text.replace("hahaha", "[Laughter]")
		text = text.replace("Ha-Ha", "[Laughter]")

		return text.strip()
	except sr.UnknownValueError:
		print("Sorry, didn't get that.") #Google Cloud Speech could not understand audio
	except sr.RequestError as e:
		print("There seems to be a problem with transcribing your speech. Error with Google Cloud Speech service; {0}".format(e))

	return ""

def predictLaughter(example, acoustic):
	X, _, _ = fitModel(example, acoustic, vocab=vocab, frequent_ngram_col_idx=freq_col_idx)
	return regr.predict(X)

def play_laughtrack():
	laughFiles = ["laughtracks/laughtrack{}.wav".format(i) for i in range(1, 8)]
	rand = np.random.randint(0,len(laughFiles))
	return_code = subprocess.call(["afplay", laughFiles[rand]])

# set up google cloud credential
with open ('service_account_key.json', 'r') as f:
	credential = f.read()

# initialize model
print("Setting up")
graph = tf.Graph()
with graph.as_default():
	model = RNNModel()
	init = tf.global_variables_initializer()
vocab = pickle.load(open(savedvocabularyfile, 'rb'))
freq_col_idx = pickle.load(open(savedfreqcolidxfile, 'rb'))
regr = pickle.load(open(savedlogmodelfile, 'rb'))

@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/record')
def record():
    print("Recording")
    global p
    global stream
    global frames
    frames = []
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
					channels=CHANNELS,
					rate=RATE,
					input=True,
					stream_callback=callback)
    return ""

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	if not stream:
		return ""
	# stop recording
	stream.stop_stream()
	stream.close()
	p.terminate()
	# save recording
	wf = wave.open(audioFile, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()

	print("Processing audio")
	convert_audio_sample()
	test_dataset = load_dataset(audioFileProcessed)
	feature_b, label_b, seqlens_b = make_batches(test_dataset, batch_size=len(test_dataset[0]))
	feature_b = pad_all_batches(feature_b)
	with tf.Session(graph=graph) as session:
		session.run(init)
		# Load pretrained model
		new_saver = tf.train.import_meta_graph('saved_models/model.meta', clear_devices=True)
		new_saver.restore(session, 'saved_models/model')
		batch_cost, summary, acc, predicted, acoustic = model.train_on_batch(session, feature_b[0], label_b[0], seqlens_b[0], train=False)
		print("Running language model")
		text = get_transcript_from_file(credential)
		print "transcript:", text
		example = [(text, 0)]
		prediction = predictLaughter(example, acoustic)[0]
		print "prediction:", prediction
		if prediction == 1:
			play_laughtrack()

 		return {"funny": prediction, "text": text}

if __name__ == "__main__":
    app.run(debug=False, port=5000)
