import numpy as np

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

    #You can also batch long audio files for speech recognition (using Asynchronous Recognition requests) or have the Speech API listen to a stream and return results while the stream is open using a gRPC StreamingRecognize request. For example, a streaming recognition task may provide transcription from a user while they are speaking. 

	#https://pypi.python.org/pypi/SpeechRecognition/
    pip install SpeechRecognition
    Google API Client Library for Python (required only if you need to use the Google Cloud Speech API, recognizer_instance.recognize_google_cloud) pip install google-api-python-client
    brew install portaudio && sudo brew link portaudio
    pip install pyaudio
'''


import speech_recognition as sr

# obtain audio from the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something and laughbot will decide whether you're funny!")
    audio = r.listen(source)

# recognize speech using Google Cloud Speech
#If credentials not specified, the library will try to automatically `find the default API credentials JSON file <https://developers.google.com/identity/protocols/application-default-credentials>`__.
#currently this default is an environment variable set as "./service_account_key.json". used to be CS224s-Laughbot-cdb7a14ba039.json
'''GOOGLE_CLOUD_SPEECH_CREDENTIALS = {}''' #If default credentials not working from the environment variable, insert credentials here and add to r.recognize_google_cloud call
try:
    print("Google Cloud Speech thinks you said " + r.recognize_google_cloud(audio))#, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS))
except sr.UnknownValueError:
    print("Google Cloud Speech could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Cloud Speech service; {0}".format(e))





