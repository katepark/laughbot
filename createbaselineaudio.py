## NOTE: MUST run on corn (too expensive to scp csv files over rip)

import random
import os 
import glob
import re
import numpy as np
import pickle

# change location of transcripts / mrk files and audio csv feature files
TRANS_PATH = '/afs/ir/data/linguistic-data/Switchboard/Switchboard-Transcripts/swb1/trans/'
CSV_PATH = '/audio'

LAUGHTER_REGEX = '[\[<][Ll]aughter[\]>].?'
HEADER_SKIP = 15 # skip over header, start at line 16
DATA_PORTION = 0.2
MAX_AUDIO_VECTOR = 50 # max # of timesteps (10ms each) of audio features to store for each line
SKIP_LIST = ['sw2155','sw2191','sw2235','sw2289','sw2298','sw2299','sw2554','sw2632','sw2644','sw3094','sw3180','sw4361','sw4379'] # missing mrk or audio files

def writeAudio(allAudio, matrix, allLabels, label, allLens):
  allAudio.append(matrix)
  allLabels.append(label)
  allLens.append(len(matrix))


def buildTrainSet():
  # raw sentences
  trainFile = open('switchboardsampleL.train', 'w')
  valFile = open('switchboardsampleL.val', 'w')
  testFile = open('switchboardsampleL.test', 'w')
  # audio features
  trainAudio = open('switchboardaudioL.train.pkl', 'wb')
  valAudio = open('switchboardaudioL.val.pkl', 'wb')
  testAudio = open('switchboardaudioL.test.pkl', 'wb')

  num_punchlines = [0]*3
  num_unpunchlines = [0]*3
  # iterate through all files in data
  trainAudioVectors = []
  trainLabels = []
  trainLens = []
  valAudioVectors = []
  valLabels = []
  valLens = []
  testAudioVectors = []
  testLabels = []
  testLens = []


  for subdir, dirs, files in os.walk(TRANS_PATH): # walks through all disc files
    for filename in files:
      base, ext = filename.split('.')
      if ext != 'mrk' or base in SKIP_LIST:
        continue
      filepath = os.path.join(subdir, base + '.txt')
      times = os.path.join(subdir, filename)
      csvfile = os.path.join(os.getcwd() + CSV_PATH, base[0:2] + '0' + base[2:]  + '.csv')
      print base
      
      alllines = open(filepath).read().splitlines()
      lines = [x for x in alllines if x != '' and not x.isspace()]
      mrk = []
      with open(times) as f:
        for line in f:
          if line != '' and not line.isspace():
            info = line.split()
            if info[0] =='@' or info[0] == 'i' or info[0] == 'a':
              info = info[1:]
            if len(info) < 4:
              continue
            if info[1][0] == '&':
              info[1] = info[1][2:]
            mrk.append(info)
      mfccs = np.genfromtxt(csvfile, delimiter=',', skip_header=1)

      start = 0
      prevAudio = []
      for i in range(HEADER_SKIP, len(lines)):
        punchLineFound = False
        line = lines[i]
        # find matching time interval in MRK
        words = line.split()
        end = len(words) + start
        if words[0][-2:] == 'A:' or words[0][-2:] == 'B:':
          end -= 1
        startMfcc = start
        while mrk[startMfcc][1] == '*':
          startMfcc += 1
        endMfcc = min(end + 1, len(mrk) - 1)
        # TODO: currently counting end as beginning of next speaker,
        # can also change so end is the actual end of this speaker
        while mrk[endMfcc][1] == '*':
          if endMfcc == len(mrk) - 1:
            break
          endMfcc += 1
        if endMfcc == len(mrk) - 1:
          continue
        interval = (int(100 * float(mrk[startMfcc][1])), int(100 * float(mrk[endMfcc][1])))
        audio = mfccs[max(interval[0], interval[1] - MAX_AUDIO_VECTOR):interval[1],     2:]
        start = end
        # classify line
        if line != '' and random.random() < DATA_PORTION:
          matches = re.finditer(LAUGHTER_REGEX, line)
          # gets corresponding audio features for interval
          # columns 0-1 track indices and are ignored
          if matches: # Laughter Found, Punchline
            for m in matches:
              if m.start(0) < 10: # ignore those who laugh at themselves
                words = line.split(' ')
                punchline = lines[i-1]
                classifiedLine = '1 ' + punchline + '\n'
                rando = random.random()
                if rando < 0.8:
                  trainFile.write(classifiedLine)
                  writeAudio(trainAudioVectors, prevAudio, trainLabels, 1, trainLens)
                  num_punchlines[0] += 1
                elif rando < 0.9:
                  valFile.write(classifiedLine)
                  writeAudio(valAudioVectors, prevAudio, valLabels, 1, valLens)
                  num_punchlines[1] += 1
                else:
                  testFile.write(classifiedLine)
                  writeAudio(testAudioVectors, prevAudio, testLabels, 1, testLens)
                  num_punchlines[2] += 1

                punchLineFound = True
                break
                # print 'PUNCHLINE', classifiedLine
          if not punchLineFound: # No Laughter Found, Unfunny line
            unpunchline = lines[i-1]
            classifiedLine = '0 ' + unpunchline + '\n'
            rando = random.random()
            if rando < 0.8:
              if random.random() < 0.05:  # sample because too many unfunny lines
                num_unpunchlines[0] += 1
                trainFile.write(classifiedLine)
                writeAudio(trainAudioVectors, prevAudio, trainLabels, 0, trainLens)
            elif rando < 0.9:
              if random.random() < 0.05:  # sample because too many unfunny lines
                num_unpunchlines[1] += 1
                valFile.write(classifiedLine)
                writeAudio(valAudioVectors, prevAudio, valLabels, 0, valLens)
            else:
              num_unpunchlines[2] += 1
              testFile.write(classifiedLine)
              writeAudio(testAudioVectors, prevAudio, testLabels, 0, testLens)
            # print 'UNFUNNY', classifiedLine
        prevAudio = audio

  # saves an audio features matrix to pickle file
  # format of output file:
  # (examples, targets, lengths) tuple
  # call pickle.load(f) to get tuple
  pickle.dump((trainAudioVectors, trainLabels, trainLens), trainAudio)
  pickle.dump((valAudioVectors, valLabels, valLens), valAudio)
  pickle.dump((testAudioVectors, testLabels, testLens), testAudio)

  for i in range(len(num_punchlines)):
    print 'STATS', 'NUM PUNCHLINES', num_punchlines[i], 'NUM UNFUNNY LINES', num_unpunchlines[i], 'Fraction of Punchlines', float(num_punchlines[i]) / (num_punchlines[i] + num_unpunchlines[i])



buildTrainSet()
