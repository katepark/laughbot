from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
import util
import random
import collections
import math
import sys
import numpy as np


def fitModel(examples, vocab=None):
        corpus = []
        for x,y in examples:
            corpus.append(x)
        vectorizer = CountVectorizer(vocabulary=vocab, ngram_range=(1, 3),token_pattern=r'\b\w+\b', min_df=1)
        X = vectorizer.fit_transform(corpus)
        # analyze = vectorizer.build_analyzer()
        fullfeature = X.toarray()
        print 'SHAPE', len(fullfeature), len(fullfeature[0])
        # return vectorizer
        return fullfeature, vectorizer.vocabulary_

def featureExtractor(punchline, vocab):
        # print vectorizer.transform(punchline).toarray()
        feature = vectorizer.transform(punchline).toarray()
        # print 'SHAPE INDIV', len(feature), len(feature[0])
        return vectorizer.transform(punchline).toarray()
    	
def learnPredictor(trainExamples, valExamples, testExamples):
    	print 'BEGIN: GENERATE TRAIN'
        trainFeatures, vocabulary = fitModel(trainExamples)
        trainX = []
    	trainY = []
    	for x,y in trainExamples:
    		# phi = featureExtractor(x, vectorizer)
            phi = trainFeatures[len(trainX)]
            trainX.append(phi)
            trainY.append(y)

    	print 'END: GENERATE TRAIN'
    	print 'BEGIN: GENERATE TEST'
        testFeatures, _ = fitModel(testExamples, vocab=vocabulary)
    	testX = []
    	testY = []
    	for x,y in testExamples:
    		phi = testFeatures[len(testX)]
    		testX.append(phi)
    		testY.append(y)
    	print 'END: GENERATE TEST'
    	# testX = np.reshape(testX, (len(testX), numFeatures))
    	# print "Size: ", len(testX), len(testX[0])

    	# # val
        
        print "BEGIN: TRAINING"
        regr = LogisticRegression()
        regr.fit(trainX, trainY)
        print "END: TRAINING"
        trainPredict = regr.predict(trainX)
        testPredict = regr.predict(testX)

        precision,recall,fscore,support = precision_recall_fscore_support(trainY, trainPredict, average='binary')
        print "LOGISTIC TRAIN scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)
        precision,recall,fscore,support = precision_recall_fscore_support(testY, testPredict, average='binary')
        print "LOGISTIC TEST scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)

def allPosNegBaseline(trainExamples, valExamples, testExamples):
    print 'ALL POSITIVE TRAIN scores:'
    allPos(trainExamples)
    print 'ALL POSITIVE TEST scores:'
    allPos(testExamples)
    print 'ALL NEGATIVE TRAIN scores:'
    allNeg(trainExamples)
    print 'ALL NEGATIVE TEST scores:'
    allNeg(testExamples)

def allPos(examples):
    num_punchlines = 0
    for x,y in examples:
        if y == 1:
            num_punchlines += 1
    precision = float(num_punchlines)/len(examples)
    recall = 1.0
    fscore = 2.0 * precision * recall / (precision + recall)

    print "\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)

def allNeg(examples):
    num_punchlines = 0
    for x,y in examples:
        if y == 1:
            num_punchlines += 1
    precision = 1.0 - float(num_punchlines)/len(examples)
    recall = 0.0
    fscore = 0.0
    print "\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)


trainExamples = util.readExamples('switchboardsample.train')
valExamples = util.readExamples('switchboardsample.val')
testExamples = util.readExamples('switchboardsample.test')
learnPredictor(trainExamples, valExamples, testExamples)
allPosNegBaseline(trainExamples, valExamples, testExamples)