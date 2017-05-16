import random
import os 
import glob

laughterregex = '[\[<][Ll]aughter[\]>].?'

def buildTrainSet():
	trainFile = open('switchboardsample.train', 'w')
	validationFile = open('switchboardsample.val', 'w')
	testFile = open('switchboardsample.test', 'w')
	print os.getcwd()
	# iterate through all files in data
	for inputname in glob.iglob(os.getcwd() + '/data/*'):
		header_found = False
		with open(inputname) as input:
			alllines = input.read().splitlines()
			lines = [x for x in alllines if x != '']

			# TODO: skip over header, start at line 18
			for i in range(18, len(lines)):
				line = lines[i]
				if line != '':
					print line 
					words = line.split(' ')
					# TODO: REGEX
					if '[Laughter]' in words or '[Laughter].' in words or '[Laughter],' in words: # [Laughter], [laughter], with punctuation
						# TODO DECIDE: Strip <laughter> from punchlines before train??
						# punchline = stripLaughter(lines[i-1].split(' '))
						# punchline = ' '.join(punchline) # prevLine is punchline
						punchline = lines[i-1]
						punchline = '1 ' + punchline + '\n'
						classifiedLine = punchline
					else:
						punchline = lines[i-1]
						classifiedLine = '0 ' + punchline + '\n'
					if random.random() < 0.8:
						trainFile.write(classifiedLine)
					elif random.random() < 0.9:
						validationFile.write(classifiedLine)
					else:
						testFile.write(classifiedLine)
					# 	print classifiedLine


def stripLaughter(words):
	strippedLine = [x for x in words if (x.find('aughter>') < 0)]
	return strippedLine


buildTrainSet()