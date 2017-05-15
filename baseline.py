from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import util
import random
import collections
import math
import sys
import numpy as np


class Regression:

	def featureExtractor(self, punchline):
	
	def learnPredictor(self, trainExamples, testExamples):
		print 'BEGIN: GENERATE TRAIN'
		trainX = []
		trainY = []
		for x,y in trainExamples:
			phi = self.featureExtractor(x)
			trainX.append(phi)
			trainY.append(y)
		print 'END: GENERATE TRAIN'
		
		print 'BEGIN: GENERATE TEST'
		testY = []
		textX = []
		for x,y in testExamples:
			phi = self.featureExtractor(x)
			testX.append(phi)
			testY.append(y)
		print 'END: GENERATE TEST'
        
        print "BEGIN: TRAINING"
		regr = LogisticRegression().fit(trainX, trainY)
        print "END: TRAINING"




trainExamples = util.readExamples('switchboardsample.train')
testExamples = util.readExamples('switchboardsample.test')