from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
import util
import random
import collections
import math
import sys
import numpy as np


ngram_threshold = 10
# 8 357143
# 10 0.482759

def fitModel(examples, vocab=None, frequent_ngram_col_idx=None):
        corpus = []
        for x,y in examples:
            corpus.append(x)
        vectorizer = CountVectorizer(vocabulary=vocab, ngram_range=(1, 3),token_pattern=r'\b\w+\b', min_df=1)
        X = vectorizer.fit_transform(corpus)
        
        # analyze = vectorizer.build_analyzer()
        fullfeature = X.toarray()

        print 'SHAPE', len(fullfeature), len(fullfeature[0])

        if not frequent_ngram_col_idx:
            frequent_ngram_col_idx = []
            for i in range(fullfeature.shape[1]):
                if sum(fullfeature[:,i]) > ngram_threshold:
                    frequent_ngram_col_idx.append(i)

        fullfeature = fullfeature[:, frequent_ngram_col_idx]

        print 'NEW SHAPE', len(fullfeature), len(fullfeature[0])
        # return vectorizer
        return fullfeature, vectorizer.vocabulary_, frequent_ngram_col_idx

    	
def learnPredictor(trainExamples, valExamples, testExamples):
    	print 'BEGIN: GENERATE TRAIN'
        trainFeatures, vocabulary, freq_col_idx = fitModel(trainExamples)
        trainX = []
    	trainY = []
    	for x,y in trainExamples:
    		# phi = featureExtractor(x, vectorizer)
            phi = trainFeatures[len(trainX)]
            trainX.append(phi)
            trainY.append(y)

    	print 'END: GENERATE TRAIN'
    	print 'BEGIN: GENERATE TEST'
        testFeatures, _, freq_col_idx_test = fitModel(testExamples, vocab=vocabulary, frequent_ngram_col_idx=freq_col_idx)
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