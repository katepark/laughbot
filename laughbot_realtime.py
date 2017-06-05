import numpy as np

#The Cloud Speech API v1 is officially released and is generally available from the https://speech.googleapis.com/v1/speech endpoint.
#Should probably move the service_account_key file (CS224s-Laughbot-cdb7a14ba039.json) to local directories instead of shared for data safety and permissions reasons, after you login
'''Run this:
	export GOOGLE_APPLICATION_CREDENTIALS=./CS224s-Laughbot-cdb7a14ba039.json #set environment variable to key path
	Follow instructions for Google cloud SDK installation: https://cloud.google.com/sdk/downloads 
	https://cloud.google.com/sdk/docs/quickstarts
	(to change compute zone: gcloud config set compute/zone NAME)
	"Created a default .boto configuration file at [/Users/nataliemuenster/.boto]. See this file and
		[https://cloud.google.com/storage/docs/gsutil/commands/config] for more
		information about configuring Google Cloud Storage.
		Your Google Cloud SDK is configured and ready to use!
		"

	gcloud auth application-default login (just hit y when it asks and it will redirect you to authenticate)
	gcloud auth activate-service-account --key-file=./CS224s-Laughbot-cdb7a14ba039.json

	Using speech API: (https://cloud.google.com/speech/docs/getting-started)
	gcloud auth application-default print-access-token:
	#ex result: ya29.ElpfBCz3lei0FaA_8OJ88fssq5qArUkdQjdqVhSNKGajTk-6J38JSKCiIWeTArj4e9eZhgHhEOcO0SS0qWkm9n21KcOnagEWWEEWy3MaH5r0wHJxqfSEfXK4kC4
	call in console:
	"curl -s -k -H "Content-Type: application/json" \
    -H "Authorization: Bearer access_token" \
    https://speech.googleapis.com/v1/speech:recognize \
    -d @sync-request.json"

    #You can also batch long audio files for speech recognition (using Asynchronous Recognition requests) or have the Speech API listen to a stream and return results while the stream is open using a gRPC StreamingRecognize request. For example, a streaming recognition task may provide transcription from a user while they are speaking. 

'''

#The Cloud Speech API v1 is officially released and is generally available from the https://speech.googleapis.com/v1/speech endpoint

from oauth2client.client import GoogleCredentials
credentials = GoogleCredentials.get_application_default()

#https://cloud.google.com/docs/authentication
def create_service():
    """Creates the service object for calling the Cloud Storage API."""
    # Construct the service object for interacting with the Cloud Storage API -
    # the 'storage' service, at version 'v1'.
    # You can browse other available api services and versions here:
    #     https://developers.google.com/api-client-library/python/apis/
    return googleapiclient.discovery.build('storage', 'v1')