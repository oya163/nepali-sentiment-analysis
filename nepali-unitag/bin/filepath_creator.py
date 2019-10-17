'''
	Stemmer + POS tagger based on 
	https://www.lancaster.ac.uk/staff/hardiea/nepali/postag.php
	
	Better to run unitag separately
	because recurring questions for overwriting files
	
	Just create path for each files
'''

import subprocess
import sys
import os
import time
from subprocess import Popen, PIPE

unitag_cmd = "unitag nepali-inst.txt "

filelist = sys.argv[1]
dir = sys.argv[2]

if os.path.exists(filelist):
	os.remove(filelist)

file_count = 0
with open(filelist, 'w', encoding='utf-8') as flist:
	for root, dirs, files in os.walk(dir):
		for f in files:
			input_file = os.path.join(root, f)
			print(input_file)
			file_count += 1
			flist.write(input_file+'\n')

#cmd = unitag_cmd + filelist + ' L'

#foo_proc = Popen([cmd], stdin=PIPE, stdout=PIPE)
#yes_proc = Popen(['yes', 'y'], stdout=foo_proc.stdin)
#foo_output = foo_proc.communicate()[0]
#yes_proc.wait() # avoid zombies

# p = Popen(cmd.split(), shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)
# newline = os.linesep # [1]
# commands = []
# for i in range(0, file_count):
	# commands.append('y')
# #commands = ['y', 'y', 'y', 'y', 'y']
# p.communicate( newline.join( commands))
# p.wait()
#p.communicate(input=b'y')
		#
		
#print(subprocess.run(cmd, input=b"y", capture_output=True))