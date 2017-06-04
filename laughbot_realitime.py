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