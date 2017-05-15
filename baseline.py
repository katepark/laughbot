from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import util
import random
import collections
import math
import sys
import numpy as np


def fitModel(trainExamples):
        corpus = []
        for x,y in trainExamples:
            corpus.append(x)

        vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
        X = vectorizer.fit_transform(corpus)
        analyze = vectorizer.build_analyzer()
        fullfeature = X.toarray()
        print 'SHAPE', len(fullfeature), len(fullfeature[0])
        return vectorizer

def featureExtractor(punchline, vectorizer):
        # print vectorizer.transform(punchline).toarray()
        feature = vectorizer.transform(punchline).toarray()
        # print 'SHAPE INDIV', len(feature), len(feature[0])
        return vectorizer.transform(punchline).toarray()
    	
def learnPredictor(trainExamples, valExamples, testExamples):
    	print 'BEGIN: GENERATE TRAIN'
        vectorizer = fitModel(trainExamples)
        trainX = []
    	trainY = []
    	for x,y in trainExamples:
    		phi = featureExtractor(x, vectorizer)
    		trainX.append(phi)
    		trainY.append(y)
    	numFeatures = len(trainX[0])
        # print trainX
        # print trainY
    	# trainX = np.reshape(trainX, (len(trainX), numFeatures))
    	# print 'END: GENERATE TRAIN'
    	# print 'BEGIN: GENERATE TEST'

    	# textX = []
    	# testY = []
    	# for x,y in testExamples:
    	# 	phi = self.featureExtractor(x)
    	# 	testX.append(phi)
    	# 	testY.append(y)
    	# print 'END: GENERATE TEST'
    	# testX = np.reshape(testX, (len(testX), numFeatures))
    	# print "Size: ", len(testX), len(testX[0])

    	# # val
        
     #    print "BEGIN: TRAINING"
        regr = LogisticRegression().fit(trainX, trainY)
     #    print "END: TRAINING"
        print "TRAIN score:", regr.score(trainX, trainY)
     #    print "TEST score:", regr.score(testX, testY)



trainExamples = util.readExamples('switchboardsample.train')
valExamples = util.readExamples('switchboardsample.val')
testExamples = util.readExamples('switchboardsample.test')
learnPredictor(trainExamples, valExamples, testExamples)