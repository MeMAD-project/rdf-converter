import argparse
import sys
from time import time
import pandas as pd
import os
from os.path import exists, isfile, isdir, split, join

from helpers.PA_Store import PA_Store
from helpers.LD_Store import LD_Store
from helpers.Yle_Store import Yle_Store


parser = argparse.ArgumentParser(description='MeMAD Converter')
parser.add_argument("-p", "--path", type=str, help="Specify the path for the file or folder to process", required=True)
parser.add_argument("-d", "--dataset", choices=['ina-ld', 'ina-pa', 'yle'], type=str, help="Specify which converter to use.", required=True)

args = parser.parse_args()
argpath = args.path
dataset = args.dataset

valid_extension = 'csv' if dataset.startswith('ina') else 'xml'
converter = {'ina-ld': LD_Store, 'ina-pa': PA_Store, 'yle': Yle_Store, }

files_to_process = []

if not exists(argpath) :
	print('Error: the provided path does not exist.')
	exit()

if isfile(argpath) and argpath[-3:] == valid_extension:
	files_to_process.append(argpath)

elif isdir(argpath):
	for file in os.listdir(argpath):
		path = join(argpath, file)
		if isfile(path) and path[-3:] == valid_extension:
			files_to_process.append(path)
	print('== ' + str(len(files_to_process)) + ' file(s) to process ==')

else:
	print('Error: the provided path is not valid.')
	exit()


for file_path in files_to_process:
	path, filename   = split(file_path)
	output_file_path = join(path, filename[:-3] + 'ttl')

	# Create an empty graph if the generated file doesn't already exist or load it otherwise
	store = converter[dataset](output_file_path)
	
	# Process the file
	try:
		tick = time()
		print('Writing metadata into file : ' + output_file_path)

		if dataset.startswith('ina'):
			dataframe = pd.read_csv(file_path, sep=';', encoding='latin-1', converters={'IdentifiantMediametrie': lambda x: str(x), 'IdentifiantImedia': lambda x: str(x)}).fillna('')
			for i, entry in dataframe.iterrows():
			    store.process_entry(entry)
			    if i > 5: break
			print(str(i-1) + ' entities processed.')

		else:
			store.process_file(file_path)

		store.save()
		tock = time()
		print('Total time: {:2.2f}s'.format(tock - tick))

	except Exception as e:
		print('An error was encounted while processing file "' + file_path + '" :')
		print(str(e))
