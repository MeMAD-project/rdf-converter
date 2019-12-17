# -*- coding: utf-8 -*-

import os
from os.path import exists, dirname, join, isfile
import pandas as pd
import numpy as np
from rdflib import Namespace, URIRef, ConjunctiveGraph, Literal
from rdflib.namespace import FOAF, DC, SKOS, RDF, XSD, DCTERMS
import urllib
import re
import json
import pickle
import unicodedata
from hashlib import sha1
import time
from tqdm import tqdm
import xml.etree.ElementTree as ET
import datetime


import argparse

parser = argparse.ArgumentParser(description='MeMAD Converter')
parser.add_argument("-p", "--path", type=str, help="Specify the path for the file or folder to process", default='data/ld') #, required=True)
parser.add_argument("-o", "--output", type=str, help="Specify the path to which the TTL output would be written.", default='data/dump/') #, required=True)
parser.add_argument("-f", "--flow_mapping", type=str, help="Specify the path to a file containing the mapping between filenames and their Flow identifier.",
 default='data/yle/file_mapping.json') #, required=True)
parser.add_argument("-k", "--keep_mappings", help="add this flag to generate CSV files for mapping Programmes to their URIs", action='store_true', default=False) #, required=True)


args = parser.parse_args()
data_path   = args.path
output_path = args.output
flow_mapping_file = args.flow_mapping
keep_mappings = args.keep_mappings

if not exists(data_path) :
	print('Error: the provided path does not exist.')
	exit()

if not exists(output_path):
    print('Creating directory :' + dirname(output_path))
    os.makedirs(dirname(output_path))

data_path = data_path + '/' if data_path[-1] != '/' else data_path
repos_to_process = sorted(os.listdir(data_path))

if repos_to_process[0][-3:] == 'csv':
	dataset_name = data_path.split('/')[-2]
	data_path = data_path[:-(len(dataset_name)+1)]
	repos_to_process = [dataset_name]
	print('Processing the "', dataset_name, '" dataset @', data_path)
else:
	print('Processing', len(repos_to_process), 'datasets..')



dfs_program = []
dfs_segment = []
for dataset in repos_to_process: # ['14-may2019']: # 
	if '.' in dataset:
	    continue
	files = os.listdir(data_path+dataset)
	for file in files:
		if file.split('.')[-1] != 'csv': break
		filepath = join(data_path+dataset, file)
		df = pd.read_csv(filepath,  encoding='latin-1', delimiter=';', low_memory=False).fillna('')
		if 'TitreSujet' in df.columns:
			dfs_segment.append(df)
		else:
			dfs_program.append(df)

df_eall, df_sall = [], [] 
if dfs_program:
	df_eall = pd.concat(dfs_program, sort=False).fillna('')
	df_eall = df_eall.replace('', "'", regex=True).replace('', "-", regex=True).replace('', "", regex=True)
if dfs_segment:
	df_sall = pd.concat(dfs_segment, sort=False).fillna('')

print(len(df_eall))
print(len(df_sall))


MeMAD   = Namespace('http://data.memad.eu/ontology#')
EBUCore = Namespace('http://www.ebu.ch/metadata/ontologies/ebucore/ebucore#')

base = 'http://data.memad.eu/'

g = ConjunctiveGraph()
radio_channels = set(['BEU', 'BFM', 'CHE', 'D8_', 'EU1', 'MUV', 'GA1', 'EU2', 'FBL', 'FCR', 
                  'FIF', 'FIT', 'FMU', 'FUN', 'MUV', 'NOS', 'NRJ' , 'RBL', 'RCL' , 'RFI',
                  'RFM', 'RIR', 'RMC', 'RT2', 'RTL', 'RT9', 'SKY', 'SUD', 'VIR'])

def save_graph(path='ld.ttl'):
    g.serialize(path, format='turtle')

def reset_graph():
    global g
    g = ConjunctiveGraph()
    g.bind('memad', MeMAD)
    g.bind('skos', SKOS)
    g.bind('ebucore', EBUCore)
    g.bind('dcterm', DCTERMS)

def add_to_graph(triplet, signal_empty_values=False):
    
    if triplet[2] and len(triplet[2]) > 0 and str(triplet[2]) != 'None': # the predicate has a non-null value
        g.add(triplet)
    elif signal_empty_values:
        print(str(triplet[0]) + '.' + str(triplet[1]) + ' was not added to the graph (empty value)')


def clean_string(s):
    """ Transforming any text strings into valid ascii slugs """
    to_dash = '\\/\',.":;[]()!? #=&$%@{«°»¿=>+*\u0019\xa0' # \u0019 is NOT utf8-frieldy 
    cleaned = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    cleaned = ''.join('-' if c in to_dash else c for c in cleaned)
    cleaned = ''.join(c if i == 0 or (c == '-' and cleaned[i-1]) != '-' else '' for i, c in enumerate(cleaned))
    cleaned = cleaned.lower().strip('-')
    return cleaned
    

def transform(field, value):
    if field == 'channel':
        channel_codes = json.load(open('mappings/ina_channel2code.json'))
        return channel_codes[value].lower()
    elif field == 'datetime':
        Y, M, D = value[:10].split('-')
        date = Y + '-' + M + '-' + D
        h, m, s = value[11:13], value[14:16], value[17:19]
        if int(h) > 23:
            h = str(int(h) - 24)
            D = str(int(D) + 1)
        return Literal(date + 'T' + h + ':' + m + ':' + s, datatype=XSD.dateTime)
    elif field == 'time':
        return Literal(value, datatype=XSD.time)
    elif field == 'duration':
        if not value: return None
        h = str(int(value / 3600))
        m = str(int((value % 3600) / 60))
        s = str(value % 60)
        t = 'PT'+h.zfill(2)+'H'+m.zfill(2)+'M'+s.zfill(2)+'S'
        return Literal(t, datatype=XSD.duration)
    elif field == 'end_datetime':
        try:
            date, duration = value
            if not duration: return None
            date = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
            duration = datetime.timedelta(seconds=int(duration))
            end_datetime = (date + duration).strftime("%Y-%m-%dT%H:%M:%S")
            return Literal(end_datetime, datatype=XSD.dateTime) 
        except Exception as e:
            print("Can't generate end time, ", str(e), '(', value, ')')
    else:
        raise Exception('No transformation defined for field ' + field + '( value ) :' + str(value))


def encode_uri(resource, data):
    if resource == 'program':
        hashed = sha1(data['id'].encode()).hexdigest() 
        source = data['source'].lower()
        parent = clean_string(data['parent'])
        return URIRef(base + source + '/' + parent + '/' + hashed)

    elif resource == 'channel':
        channel_code = transform('channel', data['name'])
        return URIRef(base + 'channel/' + channel_code)

    elif resource == 'media':
        hashed = sha1(data['id'].encode()).hexdigest() 
        return URIRef(base + 'media/' + hashed)

    elif resource == 'agent':
        agent_name_cleaned = clean_string(data['name'])
        return URIRef(base + 'agent/' + agent_name_cleaned)

    elif resource in ['timeslot', 'collection']:
        if not data['name']: return None
        name_cleaned = clean_string(data['name'])
        return URIRef(base + data['source'].lower() + '/' + name_cleaned)

    elif resource == 'history':
        return URIRef(str(data['program_uri']) + '/publication')

    elif resource == 'publication':
        # datetime = ''.join(c for c in data['datetime'] if c in '0123456789')
        n = data['n']
        return URIRef(str(data['program_uri']) + '/publication/' + n)

    else:
        raise Exception('No URI encoding for resource ' + resource)

def time_between(d1, d2):
    # "2014-05-01 05:32:32", "2014-05-01T05:33:17+01:00"
    d1 = datetime.datetime.strptime(d1, "%Y-%m-%d %H:%M:%S")
    d2 = datetime.datetime.strptime(d2[:19], "%Y-%m-%d %H:%M:%S")
    diff = max(d1, d2) - min(d1, d2)
    return (datetime.datetime.min + diff).time().strftime("%H:%M:%S")

def time_after(t, d):
    if d == str(None) : return None
    t = datetime.datetime.strptime(t, "%H:%M:%S")
    if '.' in d: d = d.split('.')[0]+'S'
    d = datetime.datetime.strptime(d, "PT%HH%MM%SS")
    
    d = datetime.timedelta(hours=d.hour, minutes=d.minute, seconds=d.second)
    return (d + t).time().strftime("%H:%M:%S")


if len(df_eall) > 0:
	reset_graph()
	mapping = []

	for i, entry in tqdm(df_eall.iterrows(), total=len(df_eall)):
	    try:
	        assert('Identifiant' in entry)  
	    except Exception:
	        raise Exception('The provided file doesn\'t have the appropriate Legal Deposit program format')

	    # Source
	    channel_name   = entry['Chaine']        
	    channel_uri    = encode_uri('channel', {'name': channel_name})
	    channel_code   = transform('channel', channel_name)
	    radio_program  = channel_code.upper() in radio_channels

	    add_to_graph((channel_uri, RDF.type, EBUCore.PublicationChannel))
	    add_to_graph((channel_uri, EBUCore.publicationChannelId, Literal(channel_code.upper())))
	    add_to_graph((channel_uri, EBUCore.publicationChannelName, Literal(channel_name)))
	    add_to_graph((channel_uri, EBUCore.serviceDescription, Literal(("Radio" if radio_program else "TV") + ' channel')))

	    
	    timeslot_name = entry['TitreTrancheHoraire']
	    timeslot_uri = encode_uri('timeslot', {'name': timeslot_name, 'source':channel_code})
	    
	    if timeslot_uri:
	        add_to_graph((timeslot_uri, RDF.type, MeMAD.Timeslot))
	        add_to_graph((timeslot_uri, EBUCore.title, Literal(timeslot_name)))

	    collection_name = entry['TitreCollection']
	    collection_uri  = encode_uri('collection', {'name': collection_name, 'source':channel_code})
	    
	    if collection_uri:
	        add_to_graph((collection_uri, RDF.type, EBUCore.Collection))
	        add_to_graph((collection_uri, EBUCore.title, Literal(collection_name)))

	    program_id  = entry['Identifiant']

	    parent = 'orphan'
	    if collection_name or timeslot_name: # if the program has a parent collection
	        parent = collection_name if collection_name else timeslot_name

	    program_uri = encode_uri('program', {'id': program_id, 'source': channel_code, 'parent': parent})
	    program_type = EBUCore.RadioProgramme if radio_program else EBUCore.TVProgramme
	    
	    if collection_uri: add_to_graph((collection_uri, EBUCore.isParentOf, program_uri))
	    if timeslot_uri  : add_to_graph((timeslot_uri, EBUCore.isParentOf, program_uri))
	        
	    
	    # Program Metadata
	    title            = entry['TitreEmission'].strip()
	    summary          = entry['Resume'].strip().replace('\r', '')
	    lead             = entry['Chapeau'].strip().replace('\r', '')
	    producer_summary = entry['ResumeProducteur'].strip().replace('\r', '')
	    duration         = transform('duration', entry['DureeSecondes'])
	    

	    add_to_graph((program_uri, DCTERMS.publisher, Literal("INA-LD")))
	    add_to_graph((program_uri, RDF.type, program_type))
	    add_to_graph((program_uri, EBUCore.hasIdentifier, Literal(program_id)))
	    add_to_graph((program_uri, EBUCore.title, Literal(title)))
	    add_to_graph((program_uri, EBUCore.summary, Literal(summary)))
	    add_to_graph((program_uri, MeMAD.producerSummary, Literal(producer_summary)))
	    add_to_graph((program_uri, MeMAD.lead, Literal(lead)))
	    add_to_graph((program_uri, EBUCore.duration, duration))

	    # Media
	    Imedia_id        = entry['IdentifiantImedia']
	    Mediametrie_id   = entry['IdentifiantMediametrie']

	    media_uri        = encode_uri('media', {'id': program_id})

	    add_to_graph((media_uri, RDF.type, EBUCore.MediaResource))
	    add_to_graph((program_uri, EBUCore.isInstantiatedBy, media_uri))

	    add_to_graph((media_uri, MeMAD.hasImediaIdentifier, Literal(Imedia_id)))
	    add_to_graph((media_uri, MeMAD.hasMediametrieIdentifier, Literal(Mediametrie_id)))

	    # Genres
	    genres     = entry['Genres'].strip().split('|')
	    for genre in genres:
	        if genre.strip() : add_to_graph((program_uri, EBUCore.hasGenre, Literal(genre)))
	            
	    # Keywords
	    keywords   = entry['Descripteurs'].strip().split('|')
	    for keyword in keywords:
	        if keyword.strip() : add_to_graph((program_uri, EBUCore.hasKeyword, Literal(keyword)))

	    # Producers
	    producers  = entry['Producteurs'].strip().split('|')
	    for producer in producers:
	        if producer.strip() : add_to_graph((program_uri, EBUCore.hasProducer, Literal(producer)))
	    
	    # Themes
	    themes     = entry['Thematique'].strip().split('|')
	    for theme in themes:
	        if theme.strip() : add_to_graph((program_uri, EBUCore.hasTheme, Literal(theme)))
	    
	    # Contributors
	    credits    = entry['Generiques'].strip().split('|')
	    for credit in credits:
	        if credit == '': continue
	        if '#' in credit : uid, credit = credit.split('#')
	        if '(' in credit : name, role = credit.split('(')

	        role = role.strip()[:-1] # remove ')'
	        name = name.strip()
	        agent_uri = encode_uri('agent', {'name': name})

	        add_to_graph((agent_uri, RDF.type, EBUCore.Agent))
	        add_to_graph((agent_uri, EBUCore.agentName, Literal(name)))
	        add_to_graph((agent_uri, EBUCore.hasRole, Literal(role)))
	        add_to_graph((program_uri, EBUCore.hasContributor, agent_uri))

	    # Pubevent
	    pubevent_datetime     = transform('datetime', entry['startDate'])
	    pubevent_datetime_end = transform('datetime', entry['endDate'])

	    history_uri = encode_uri('history', {'program_uri': program_uri})
	    pubevent_uri = encode_uri('publication', {'program_uri': program_uri, 'n': '0'})

	    add_to_graph((history_uri, RDF.type, EBUCore.PublicationHistory))
	    add_to_graph((program_uri, EBUCore.hasPublicationHistory, history_uri))
	    add_to_graph((history_uri, EBUCore.hasPublicationEvent, pubevent_uri))

	    add_to_graph((pubevent_uri, RDF.type, EBUCore.PublicationEvent))
	    add_to_graph((pubevent_uri, RDF.type, MeMAD.FirstRun))
	    add_to_graph((pubevent_uri, EBUCore.publishes, program_uri))
	    add_to_graph((pubevent_uri, EBUCore.isReleasedBy, channel_uri))
	    add_to_graph((pubevent_uri, EBUCore.publicationStartDateTime, pubevent_datetime))
	    add_to_graph((pubevent_uri, EBUCore.publicationEndDateTime, pubevent_datetime_end))
	    
	    mapping.append((program_id, str(program_uri), channel_code, str(pubevent_datetime), str(pubevent_datetime_end)))


	print('Serializing the graph ..')
	tick = time.time()
	save_graph(path=output_path + 'ld.ttl')
	print('Time elapsed:', round(time.time() - tick, 2), "seconds")

	mapping_df = pd.DataFrame(mapping, columns=['identifier', 'URI', 'channel', 'start', 'end'])
	mapping_df.to_csv('ina_ld_mapping.csv', index=False, encoding='utf-8')


if len(df_sall) > 0:
	reset_graph()
	segment_mapping = []

	for i, entry in tqdm(df_sall.iterrows(), total=len(df_sall)):

	    try:
	        assert('Identifiant' in entry)  
	    except Exception:
	        raise Exception('The provided file doesn\'t have the appropriate Legal Deposit program format')

	    # Source
	    channel_name   = entry['Chaine']        
	    channel_uri    = encode_uri('channel', {'name': channel_name})
	    channel_code   = transform('channel', channel_name)
	    radio_program  = channel_code in radio_channels

	    add_to_graph((channel_uri, RDF.type, EBUCore.PublicationChannel))
	    add_to_graph((channel_uri, EBUCore.publicationChannelId, Literal(channel_code.upper())))
	    add_to_graph((channel_uri, EBUCore.publicationChannelName, Literal(channel_name)))
	    add_to_graph((channel_uri, EBUCore.serviceDescription, Literal(("Radio" if radio_program else "TV") + ' channel')))

	    
	    timeslot_name = entry['TitreTrancheHoraire']
	    timeslot_uri = encode_uri('timeslot', {'name': timeslot_name, 'source':channel_code})
	    
	    if timeslot_uri:
	        add_to_graph((timeslot_uri, RDF.type, MeMAD.Timeslot))
	        add_to_graph((timeslot_uri, EBUCore.title, Literal(timeslot_name)))

	    collection_name = entry['TitreCollection']
	    collection_uri  = encode_uri('collection', {'name': collection_name, 'source':channel_code})
	    
	    if collection_uri:
	        add_to_graph((collection_uri, RDF.type, EBUCore.Collection))
	        add_to_graph((collection_uri, EBUCore.title, Literal(collection_name)))

	    segment_id  = entry['Identifiant']

	    parent = 'orphan'
	    if collection_name or timeslot_name: # if the program has a parent collection
	        parent = collection_name if collection_name else timeslot_name
	    
	    program_uri = encode_uri('program', {'id': segment_id[:-4], 'source': channel_code, 'parent': parent})
	    segment_uri = encode_uri('program', {'id': segment_id, 'source': channel_code, 'parent': parent})

	    
	    # Program Metadata
	    title            = entry['TitreEmission'].strip()
	    lead             = entry['Chapeau'].strip().replace('\r', '')
	    duration         = transform('duration', entry['DureeSecondes'])

	    add_to_graph((segment_uri, RDF.type, EBUCore.Part))
	    add_to_graph((segment_uri, EBUCore.hasIdentifier, Literal(segment_id)))
	    add_to_graph((program_uri, EBUCore.hasPart, segment_uri))
	    add_to_graph((segment_uri, EBUCore.title, Literal(title)))
	    add_to_graph((segment_uri, MeMAD.lead, Literal(lead)))
	    add_to_graph((segment_uri, EBUCore.duration, duration))

	    
	    # Keywords
	    keywords   = entry['Descripteurs'].strip().split('|')
	    for keyword in keywords:
	        if keyword.strip() : add_to_graph((segment_uri, EBUCore.hasKeyword, Literal(keyword)))

	    # Contributors
	    credits    = entry['Generique'].strip().split('|')
	    for credit in credits:
	        if credit == '': continue
	        if '#' in credit : uid, credit = credit.split('#')
	        if '(' in credit : name, role = credit.split('(')

	        role = role.strip()[:-1] # remove ')'
	        name = name.strip()
	        agent_uri = encode_uri('agent', {'name': name})

	        add_to_graph((agent_uri, RDF.type, EBUCore.Agent))
	        add_to_graph((agent_uri, EBUCore.agentName, Literal(name)))
	        add_to_graph((agent_uri, EBUCore.hasRole, Literal(role)))
	        add_to_graph((segment_uri, EBUCore.hasContributor, agent_uri))

	    # Pubevent
	    pubevent_datetime     = transform('datetime', entry['startDate'])
	    pubevent_end_datetime = transform('end_datetime', (str(pubevent_datetime), entry['DureeSecondes']))
	    """    
	    history_uri = encode_uri('history', {'program_uri': segment_uri})
	    pubevent_uri = encode_uri('publication', {'program_uri': segment_uri, 'n': '0'})

	    add_to_graph((history_uri, RDF.type, EBUCore.PublicationHistory))
	    add_to_graph((segment_uri, EBUCore.hasPublicationHistory, history_uri))
	    add_to_graph((history_uri, EBUCore.hasPublicationEvent, pubevent_uri))

	    add_to_graph((pubevent_uri, RDF.type, EBUCore.PublicationEvent))
	    add_to_graph((pubevent_uri, EBUCore.publishes, program_uri))
	    add_to_graph((pubevent_uri, EBUCore.isReleasedBy, channel_uri))
	    add_to_graph((pubevent_uri, EBUCore.hasPublicationStartDateTime, pubevent_datetime))
	    add_to_graph((pubevent_uri, EBUCore.hasPublicationEndDateTime, pubevent_end_datetime))
	    """
	    source_program = df_eall[df_eall['Identifiant'] == segment_id[:-4]]
	    source_program_start = source_program['startDate'].iloc[0]
	    segment_start = str(transform('datetime', entry['startDate'])).replace('T', ' ')
	    start = time_between(source_program_start, segment_start)
	    t_start = transform('time', start)
	    end = time_after(start, str(duration))
	    t_end = transform('time', end)

	    add_to_graph((segment_uri, EBUCore.start, t_start))
	    add_to_graph((segment_uri, EBUCore.end, t_end))

	    segment_mapping.append((segment_id, str(segment_uri), channel_code, str(pubevent_datetime), str(pubevent_end_datetime)))


	print('Serializing the graph ..')
	tick = time.time()
	save_graph(path=output_path + 'ld_sujets.ttl')
	print('Time elapsed:', round(time.time() - tick, 2), "seconds")

	segment_mapping_df = pd.DataFrame(segment_mapping, columns=['identifier', 'URI', 'channel', 'start', 'end'])
	segment_mapping_df.to_csv('ina_ld_segments_mapping.csv', index=False)





if flow_mapping_file:
	print('FLOW triplets generation..')

	data = json.load(open(flow_mapping_file, 'r'))


	reset_graph()
	found = []
	mapping_all = []

	for obj in data:
	    try:
	        filename = obj['name']
	        identifier = filename.split('.')[0][1:] if filename.startswith('R') else filename.split('.')[0]
	        
	        try:
	            program     = mapping_df[mapping_df['identifier'] == identifier].iloc[0]
	        except:
	            program     = mapping_df[mapping_df['identifier'] == 'R'+identifier].iloc[0]
	        
	        program_uri = program['URI']
	        media_uri   = URIRef(base + 'media/' + program_uri.split('/')[-1])
	        flow_href   = URIRef(obj['flowHRef'])
	        
	        add_to_graph((media_uri, EBUCore.locator, flow_href))
	        add_to_graph((media_uri, EBUCore.filename, Literal(obj['name'])))
	        found.append(identifier)
	        mapping_all.append({'uri':str(media_uri), 'flow_href': str(flow_href), 'filename': filename})
	        
	    except Exception as e:
	        pass

	save_graph(path=output_path+'ld_flow_filenames.ttl')
	mapping_all_df = pd.DataFrame(mapping_all)
	mapping_all_df.to_csv('ina_ld_flow_mapping.csv')

	print('INA LD Flow mappings have been succesfully generated.')



if not keep_mappings:
	print('Deleting mappings.. Done.')
	os.remove("ina_ld_mapping.csv")
	os.remove("ina_ld_segments_mapping.csv")
	os.remove("ina_ld_flow_mapping.csv")

