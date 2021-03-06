from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import util
import random
import collections
import math
import sys
import numpy as np
from sklearn.metrics import classification_report

ngram_threshold = 7
# 5 .432
# 6 .424

def fitModel(examples, vocab=None, frequent_ngram_col_idx=None):
        corpus = []
        for x,y in examples:
            corpus.append(x)
            print x
        vectorizer = CountVectorizer(vocabulary=vocab, ngram_range=(1, 3),token_pattern=r'\b\w+\b', min_df=1)
        X = vectorizer.fit_transform(corpus)
        
        # analyze = vectorizer.build_analyzer()
        fullfeature = X.toarray()
        
        print 'SHAPE', len(fullfeature), len(fullfeature[0])

        # The most time expensive part (pruning so only frequent ngrams used)
        '''
        if not frequent_ngram_col_idx:
            frequent_ngram_col_idx = []
            for i in range(fullfeature.shape[1]):
                if sum(fullfeature[:,i]) > ngram_threshold:
                    frequent_ngram_col_idx.append(i)

        fullfeature = fullfeature[:, frequent_ngram_col_idx]
        print 'NEW SHAPE', len(fullfeature), len(fullfeature[0])
        '''
        #Add features from grammatical context in transcript

        fullfeature = contextualFeatures(examples, fullfeature)

        print 'CONTEXTUAL SHAPE', len(fullfeature), len(fullfeature[0])
        # return vectorizer
        return fullfeature, vectorizer.vocabulary_, frequent_ngram_col_idx


#http://www.nltk.org/book/ch06.html
#Cite SentiWordNet for positivity, negativity, objectivity (http://sentiwordnet.isti.cnr.it/)
def contextualFeatures(examples, fullfeature):
    add_features = np.zeros((len(fullfeature), 8)) #4 new features added (pos has 4 elems)
    sid = SentimentIntensityAnalyzer()
    for line in xrange(len(examples)): 
        #features added: pos, punchline_len, avg_word_len, sentiment
        
        punchline = examples[line][0].split()
        add_features[line][:5] = extract_pos(punchline) #parts of speech (pos)
        add_features[line][5] = len(punchline) #punchline_len
        tot_word_length = 0
        for word in punchline:
            tot_word_length += len(punchline)
        avg_word_len = tot_word_length/len(punchline) if len(punchline) != 0 else 0
        add_features[line][6] = avg_word_len #avg_word_len

        ss = sid.polarity_scores(examples[line][0])
        add_features[line][7], ss["compound"] #sentiment

    fullfeature = np.hstack((fullfeature, add_features))
    return fullfeature

def extract_pos(punchline): #parts of speech
        tags = pos_tag(punchline)
        noun = 0
        verb = 0
        pron = 0
        adj = 0
        adv = 0
        for tag in tags:
            if tag[1] == "NN": noun += 1
            elif tag[1][:2] == "VB": verb += 1
            elif tag[1] == "JJ": adj += 1
            elif tag[1] == "RB": adv += 1
            elif tag[0].lower() == "you" or tag[0].lower() == "me" or tag[0].lower() == "I" or tag[0].lower() == "he" or tag[0].lower() == "she" or tag[0].lower() == "him" or tag[0].lower() == "her" or tag[0].lower() == "they":
                pron += 1
        return np.array([noun, verb, pron, adj, adv])

    	
def learnPredictor(trainExamples, devExamples, testExamples):
    	print 'BEGIN: GENERATE TRAIN'
        trainFeatures, vocabulary, freq_col_idx = fitModel(trainExamples)
        trainX = trainFeatures
    	trainY = [y for x,y in trainExamples]

    	print 'END: GENERATE TRAIN'
    	
        print 'BEGIN: GENERATE DEV'
        devFeatures, _, freq_col_idx_dev = fitModel(devExamples, vocab=vocabulary, frequent_ngram_col_idx=freq_col_idx)
        devX = devFeatures
        devY = [y for x,y in devExamples]
        print 'END: GENERATE DEV'
        
        print 'BEGIN: GENERATE TEST'
        testFeatures, _, freq_col_idx_test = fitModel(testExamples, vocab=vocabulary, frequent_ngram_col_idx=freq_col_idx)
        testX = testFeatures
        testY = [y for x,y in testExamples]
    	print 'END: GENERATE TEST'
        
        print "BEGIN: TRAINING"
        regr = LogisticRegression()
        regr.fit(trainX, trainY)
        print "END: TRAINING"
        trainPredict = regr.predict(trainX)
        devPredict = regr.predict(devX)
        testPredict = regr.predict(testX)
        precision,recall,fscore,support = precision_recall_fscore_support(trainY, trainPredict, average='binary')
        print "LOGISTIC TRAIN scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)
        
        precision,recall,fscore,support = precision_recall_fscore_support(devY, devPredict, average='binary')
        print "LOGISTIC DEV scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)

        precision,recall,fscore,support = precision_recall_fscore_support(testY, testPredict, average='binary')
        print "LOGISTIC TEST scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)
        return vocabulary, freq_col_idx, regr

def allPosNegBaseline(trainExamples, devExamples, testExamples):
    print 'ALL POSITIVE TRAIN scores:'
    allPos(trainExamples)
    print 'ALL POSITIVE DEV scores:'
    allPos(devExamples)    
    print 'ALL POSITIVE TEST scores:'
    allPos(testExamples)
    print 'ALL NEGATIVE TRAIN scores:'
    allNeg(trainExamples)
    print 'ALL NEGATIVE DEV scores:'
    allNeg(devExamples)
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

def realtimePredict(vocabulary, freq_col_idx, regr):
    '''
        Predicts based on inputed transcript
    '''
    x = raw_input('Give me a punchline: ')
    print x
    while (x):
        examples = []
        examples.append((x, 0))
        feature, _, _ = fitModel(examples, vocab=vocabulary, frequent_ngram_col_idx=freq_col_idx)
        predict = regr.predict(feature)
        print 'Your punchline was funny: ', predict[0]
        x = raw_input('Give me a punchline: ')


trainExamples = util.readExamples('switchboardsampleL.train')
valExamples = util.readExamples('switchboardsampleL.val')
testExamples = util.readExamples('switchboardsampleL.test')
vocabulary, freq_col_idx, regr = learnPredictor(trainExamples, valExamples, testExamples)
allPosNegBaseline(trainExamples, valExamples, testExamples)
realtimePredict(vocabulary, freq_col_idx, regr)