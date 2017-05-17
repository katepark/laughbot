import random
import os 
import glob
import re

laughterregex = '[\[<][Ll]aughter[\]>].?'

def buildTrainSet():
	trainFile = open('switchboardsample.train', 'w')
	validationFile = open('switchboardsample.val', 'w')
	testFile = open('switchboardsample.test', 'w')
	num_punchlines = 0
	num_unpunchlines = 0
	# iterate through all files in data

	for subdir, dirs, files in os.walk(os.getcwd()+ '/data/'): # walks through all disc files
		for filename in files:
			print os.path.join(subdir, filename)
			filepath = os.path.join(subdir, filename)
			with open(filepath) as input:
				alllines = input.read().splitlines()
				lines = [x for x in alllines if x != '']
				# TODO: skip over header, start at line 18
				for i in range(18, len(lines)):
					line = lines[i]
					if line != '':
						matches = re.findall(laughterregex, line)
						if matches: # Laughter Found, Punchline
							words = line.split(' ')
							# TODO DECIDE: Strip <laughter> from punchlines before train??
							# punchline = stripLaughter(lines[i-1].split(' '))
							# punchline = ' '.join(punchline) # prevLine is punchline
							punchline = lines[i-1]
							classifiedLine = '1 ' + punchline + '\n'
							num_punchlines += 1
							if random.random() < 0.8:
								trainFile.write(classifiedLine)
							elif random.random() < 0.9:
								validationFile.write(classifiedLine)
							else:
								testFile.write(classifiedLine)
						else: # No Laughter Found, Unfunny line
							unpunchline = lines[i-1]
							classifiedLine = '0 ' + unpunchline + '\n'
							if random.random() < 0.1:  # sample because too many unfunny lines
								num_unpunchlines += 1
								if random.random() < 0.8:
									trainFile.write(classifiedLine)
								elif random.random() < 0.9:
									validationFile.write(classifiedLine)
								else:
									testFile.write(classifiedLine)

	print 'STATS', 'NUM PUNCHLINES', num_punchlines, 'NUM UNFUNNY LINES', num_unpunchlines, 'Fraction of Punchlines', float(num_punchlines) / (num_punchlines + num_unpunchlines)



def stripLaughter(words):
	strippedLine = [x for x in words if (x.find('aughter>') < 0)]
	return strippedLine


buildTrainSet()