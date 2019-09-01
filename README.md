# laughbot
Predicting Humor in Spoken NLP

# Training
* If no .test/val/train files, run
`python createbaselinetrain.py`
* To run regression model:
`python baseline.py`

# Realtime
Run `python laughbot_realtime.py`

# Using Text-to-Speech transcription:
## Setup Google Cloud API
* Follow https://cloud.google.com/speech-to-text/docs/quickstart-protocol
* Create a service account, save *locally only* as [NAME].json
* Run `export GOOGLE_APPLICATION_CREDENTIALS=./[NAME].json`
  * you will need to do this everytime you open a new shell session
* Install Google Cloud SDK: https://cloud.google.com/sdk/downloads 
	* when it asks to create bash profile and leave blank or provide path, use the path it suggests but change .bash_profile to .profile (https://stackoverflow.com/questions/31084458/installed-google-cloud-sdk-but-cant-access-gcloud)
* Run `gcloud auth application-default login`
  * Login to google cloud console
* Run `gcloud auth activate-service-account --key-file=./service_account_key.json`
	* enable service account

### Details
* Commands will reference project `laughbot` by default
* Compute Engine commands will use region `us-west1` by default
* Compute Engine commands will use zone `us-west1-b` by default

## Imports
* `pip install SpeechRecognition` (https://pypi.python.org/pypi/SpeechRecognition/)
* `pip install google-api-python-client`
* `pip install gcloud`
* `brew install portaudio && brew link portaudio`
* `pip install pyaudio`
