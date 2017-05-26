import random
import os 
import glob
import re

laughterregex = '[\[<][Ll]aughter[\]>].?'
data_portion = 0.5

def buildTrainSet():
	trainFile = open('switchboardsampleL.train', 'w')
	validationFile = open('switchboardsampleL.val', 'w')
	testFile = open('switchboardsampleL.test', 'w')
	
	num_punchlines = [0]*3
	num_unpunchlines = [0]*3
	# iterate through all files in data

	for subdir, dirs, files in os.walk(os.getcwd()+ '/data/'): # walks through all disc files
		for filename in files:
			print os.path.join(subdir, filename)
			filepath = os.path.join(subdir, filename)
			with open(filepath, 'rb') as input:
				alllines = input.read().splitlines()
				lines = [x for x in alllines if x != '']
				# TODO: skip over header, start at line 18
				for i in range(18, len(lines)):
					punchLineFound = False
					line = lines[i]
					# as we do our model, only use part of data for speed
					if line != '' and random.random() < data_portion:
						matches = re.finditer(laughterregex, line)
						if matches: # Laughter Found, Punchline
							for m in matches:
								if m.start(0) < 10: 	# ignore those who laugh at themselves by only considering laughter at beginning of line
									words = line.split(' ')
									punchline = lines[i-1]
									classifiedLine = '1 ' + punchline + '\n'
									rando = random.random()
									if rando < 0.8:
										trainFile.write(classifiedLine)
										num_punchlines[0] += 1
									elif rando < 0.9:
										validationFile.write(classifiedLine)
										num_punchlines[1] += 1
									else:
										testFile.write(classifiedLine)
										num_punchlines[2] += 1

									punchLineFound = True
									# print 'PUNCHLINE', classifiedLine
						if not punchLineFound: # No Laughter Found, Unfunny line
							unpunchline = lines[i-1]
							classifiedLine = '0 ' + unpunchline + '\n'
							rando = random.random()
							if rando < 0.8:
								if random.random() < 0.05:  # sample because too many unfunny lines
									num_unpunchlines[0] += 1
									trainFile.write(classifiedLine)
							elif rando < 0.9:
								if random.random() < 0.05:  # sample because too many unfunny lines
									num_unpunchlines[1] += 1
									validationFile.write(classifiedLine)
							else:
								num_unpunchlines[2] += 1
								testFile.write(classifiedLine)
							# print 'UNFUNNY', classifiedLine

	for i in range(len(num_punchlines)):
		print 'STATS', 'NUM PUNCHLINES', num_punchlines[i], 'NUM UNFUNNY LINES', num_unpunchlines[i], 'Fraction of Punchlines', float(num_punchlines[i]) / (num_punchlines[i] + num_unpunchlines[i])



buildTrainSet()