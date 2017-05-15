import random
import os 
import glob

def buildTrainSet():
	trainFile = open('switchboardsample.train', 'w')
	validationFile = open('switchboardsample.val', 'w')
	testFile = open('switchboardsample.test', 'w')
	print os.getcwd()
	# iterate through all files in data
	for inputname in glob.iglob(os.getcwd() + '/data/*'):
		with open(inputname) as input:
			lines = input.readlines()
			for i in range(1, len(lines)):
				line = lines[i]
				words = line.split(' ')
				if '<Laughter>' in words:
					# TODO DECIDE: Strip <laughter> from punchlines before train??
					# punchline = stripLaughter(lines[i-1].split(' '))
					# punchline = ' '.join(punchline) # prevLine is punchline
					punchline = '1 ' + lines[i-1]
					classifiedLine = punchline
				else:
					classifiedLine = '0 ' + lines[i-1]
				if random.random() < 0.8:
					trainFile.write(classifiedLine)
				elif random.random() < 0.9:
					validationFile.write(classifiedLine)
				else:
					testFile.write(classifiedLine)
				print classifiedLine


def stripLaughter(words):
	strippedLine = [x for x in words if (x.find('aughter>') < 0)]
	return strippedLine


buildTrainSet()