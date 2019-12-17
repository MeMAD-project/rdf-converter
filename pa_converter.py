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
import datetime


import argparse

parser = argparse.ArgumentParser(description='MeMAD Converter')
parser.add_argument("-p", "--path", type=str, help="Specify the path for the file or folder to process", default='data/pa/') #, required=True)
parser.add_argument("-o", "--output", type=str, help="Specify the path to which the TTL output would be written.", default='data/dump/')
parser.add_argument("-s", "--subtitles", type=str, help="Specify the path to the subtitles folder.", default='data/new_ina_asr/')
parser.add_argument("-f", "--flow_mapping", type=str, help="Specify the path to the mapping between filenames and their Flow.", default='data/yle/file_mapping.json') 
parser.add_argument("-k", "--keep_mappings", help="add this flag to generate CSV files for mapping Programmes to their URIs", action='store_true', default=False) 


args = parser.parse_args()
data_path         = args.path
output_path       = args.output
subtitles_path    = args.subtitles
flow_mapping_file = args.flow_mapping
keep_mappings     = args.keep_mappings

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



MeMAD   = Namespace('http://data.memad.eu/ontology#')
EBUCore = Namespace('http://www.ebu.ch/metadata/ontologies/ebucore/ebucore#')

base = 'http://data.memad.eu/'


def extract_time(df):
    times = []
    for i, entry in df.iterrows():
        if entry['Heure de diffusion']:
            times.append(entry['Heure de diffusion'])
        else:
            diff = entry['Diffusion (aff.)']
            if '-heure:' not in diff:
                h = '00:00:00'
            else:
                _, h = diff.split('-heure:')
                broadcast_time = h[:8]
            times.append(broadcast_time)
    return times


def save_graph(path=output_path+'pa.ttl'):
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
    to_dash = '\\/\',.":;[]()!? #=&$%@{«°»¿=>+*\xa0'
    cleaned = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    cleaned = ''.join('-' if c in to_dash else c for c in cleaned)
    cleaned = ''.join(c if i == 0 or (c == '-' and cleaned[i-1]) != '-' else '' for i, c in enumerate(cleaned))
    cleaned = cleaned.lower().strip('-')
    return cleaned


def transform(field, value):
    if field == 'duration':
        if not len(value): return None
        h, m, s = value.split(':')
        value = 'PT'+h+'H'+m+'M'+s[:2]+'S'
        return Literal(value, datatype=XSD.duration)
    elif field == 'channel':
        channel_codes = json.load(open('mappings/ina_channel2code.json'))
        return channel_codes[value].lower()
    elif field == 'datetime':
        D, M, Y = value[:10].split('/')
        date = Y + '-' + M + '-' + D
        time = value[10:] 
        return Literal(date + ('T' if time else '') + time, datatype=XSD.dateTime)
    elif field == 'time':
        return Literal(value, datatype=XSD.time)
    elif field == 'date':        
        if not len(value): return None
        D, M, Y = value.split('/')
        date = Y + '-' + M + '-' + D
        return Literal(date, datatype=XSD.date)
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

    elif resource == 'record':
        return URIRef(str(data['program_uri']) + '/record')

    else:
        raise Exception('No URI encoding for resource ' + resource)


def time_between(d1, d2):
    d1 = datetime.datetime.strptime(d1, "%H:%M:%S")
    d2 = datetime.datetime.strptime(d2, "%H:%M:%S")
    diff = max(d1, d2) - min(d1, d2)
    return (datetime.datetime.min + diff).time().strftime("%H:%M:%S")


def time_after(t, d):
    t = datetime.datetime.strptime(t, "%H:%M:%S")
    d = datetime.datetime.strptime(d, "%H:%M:%S")
    
    d = datetime.timedelta(hours=d.hour, minutes=d.minute, seconds=d.second)
    return (d + t).time().strftime("%H:%M:%S")


def seconds_to_time(t, unit = 'seconds'):
    if unit == 'milliseconds':
        return str(datetime.timedelta(milliseconds=t))
    return str(datetime.timedelta(seconds=t))


def time_to_seconds(t, unit = 'seconds'):
    try:
        if t == 'None': return 0
        if 'T' in t:    t = t.split('T')[1]
        if 'day' in t:  t = t[7:]
        x = time.strptime(t.split('.')[0], '%H:%M:%S')
        s = int(datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds())
        return s
    except Exception as e:
        print("EXCEPTION", str(e), ', t = ', t)
        return 0

def preprocess_content(t):
    vowels = ['A', 'o', '1', 'Y', 'O', 'à', 'é', 'e', 'É', 'I', 'H', 'u', 'â', 'i', 'y', 'ô', 'E', 'è','a', 'U', 'ê', 'h']
    for v in vowels: t = t.replace('\' '+v, '\''+v)
    t = t.replace('- ', '-')
    t = t.replace(' -', ' ')
    
    return t

g = ConjunctiveGraph()
reset_graph()

mapping = []


dfs = []
for dataset in repos_to_process: # ['14-may2019']: # 
	if '.' in dataset:
	    continue
	files = os.listdir(data_path+dataset)
	for file in files:
		if file.split('.')[-1] != 'csv': break
		filepath = join(data_path+dataset, file)
		df = pd.read_csv(filepath,  encoding='latin-1', delimiter=';').fillna('')
		dfs.append(df)

df_all = pd.concat(dfs, sort=False).fillna('')
df_all['Heure de diffusion 2'] = extract_time(df_all)

for i, entry in tqdm(df_all.iterrows(), total=len(df_all)):

    try:
        assert('Identifiant de la notice' in entry)  
    except Exception:
        raise Exception('The provided file doesn\'t have the appropriate Professional Archive format')

    radio_program = entry['Canal de diffusion'] in ['France Inter', 'France Culture']


    # Source
    channel_name = entry['Canal de diffusion']
    channel_uri  = encode_uri('channel', {'name': channel_name})
    channel_code = transform('channel', channel_name)

    add_to_graph((channel_uri, RDF.type, EBUCore.PublicationChannel))
    add_to_graph((channel_uri, EBUCore.publicationChannelId, Literal(channel_code.upper())))
    add_to_graph((channel_uri, EBUCore.publicationChannelName, Literal(channel_name)))
    add_to_graph((channel_uri, EBUCore.serviceDescription, Literal(("Radio" if radio_program else "TV") + ' channel')))
    

    timeslot_name = entry['Titre tranche horaire'] 
    timeslot_uri  = encode_uri('timeslot', {'name': timeslot_name, 'source':channel_code}) 

    if timeslot_uri:
        add_to_graph((timeslot_uri, RDF.type, MeMAD.Timeslot))
        add_to_graph((timeslot_uri, EBUCore.title, Literal(timeslot_name)))        

    collection_name = entry['Titre collection']
    collection_uri = encode_uri('collection', {'name': collection_name, 'source':channel_code})
    if collection_uri:
        add_to_graph((collection_uri, RDF.type, EBUCore.Collection))
        add_to_graph((collection_uri, EBUCore.title, Literal(collection_name))) 

    program_id  = entry['Identifiant de la notice']

    parent = 'orphan'
    if collection_name or timeslot_name: # if the program has a parent collection
        parent = collection_name if collection_name else timeslot_name

    program_id_2 = program_id[1:] if program_id.startswith('R') else program_id
    program_uri = encode_uri('program', {'id': program_id_2, 'source': channel_code, 'parent': parent})
    mapping.append((program_id, str(program_uri)))
    

    if program_id.count('_') == 2: # this entry is a segment of a program
        source_program_uri = encode_uri('program', {'id': program_id[:-4], 'source': channel_code, 'parent': parent})
        add_to_graph((program_uri, RDF.type, EBUCore.Part))
        add_to_graph((source_program_uri, EBUCore.hasPart, program_uri))
    else:
        program_type = EBUCore.RadioProgramme if radio_program else EBUCore.TVProgramme
        add_to_graph((program_uri, RDF.type, program_type))
        
        if collection_uri: add_to_graph((collection_uri, EBUCore.isParentOf, program_uri))
        if timeslot_uri  : add_to_graph((timeslot_uri, EBUCore.isParentOf, program_uri))
    
        

    # common metadata fields to Radio and TV
    title          = entry['Titre propre'].strip()
    summary        = entry['Résumé'].strip().replace('\r', '')
    notes          = entry['Notes'].strip()
    legal_notes    = entry['Notes juridiques'].strip().replace('\r', '')
    title_notes    = entry['Notes du titre '].strip().replace('\r', '')
    corpus         = entry['Corpus  (Aff.)'].strip().replace('\r', '')
    sequences      = entry['Séquences'].strip().replace('\r', '')
    broadcasting   = entry['Type de date '] if 'Type de date' not in entry else entry['Type de date']
    duration       = entry['Durée']


    add_to_graph((program_uri, DCTERMS.publisher, Literal("INA-PA")))
    add_to_graph((program_uri, EBUCore.hasIdentifier, Literal(program_id)))
    add_to_graph((program_uri, EBUCore.hasIdentifier, Literal(program_id_2)))
    add_to_graph((program_uri, EBUCore.title, Literal(title)))
    add_to_graph((program_uri, EBUCore.summary, Literal(summary)))
    add_to_graph((program_uri, EBUCore.duration, transform('duration', duration)))
    add_to_graph((program_uri, MeMAD.titleNotes, Literal(title_notes)))
    add_to_graph((program_uri, MeMAD.corpus, Literal(corpus)))
    add_to_graph((program_uri, SKOS.note, Literal(('[Notes] ' + notes) if notes else None)))
    add_to_graph((program_uri, SKOS.note, Literal(('[Legal Notes] ' + legal_notes) if legal_notes else None)))
    add_to_graph((program_uri, MeMAD.sequence, Literal(sequences)))
    add_to_graph((program_uri, MeMAD.broadcasting, Literal(broadcasting)))


    # Radio-only metadata
    lead             = entry['Chapeau'].strip()
    recording_date   = transform('date', entry['Date d\'enregistrement'])
    producer_summary = entry['Résumé producteur'].strip()

    # TV-only metadata
    isan_number      = entry['Numéro ISAN'].strip()

    add_to_graph((program_uri, MeMAD.lead, Literal(lead)))
    add_to_graph((program_uri, EBUCore.dateCreated, Literal(recording_date)))
    add_to_graph((program_uri, MeMAD.producerSummary, Literal(producer_summary)))
    add_to_graph((program_uri, MeMAD.hasISANIdentifier, Literal(isan_number)))


    # notice
    record_creation_date = entry['Date de création']
    record_update_date   = entry['Date de modification']
    record_language      = entry['Langue de la notice']
    record_type          = entry['Type de notice']

    t_creation_date = transform('date', record_creation_date)
    t_update_date   = transform('date', record_update_date)

    record_uri = encode_uri('record', {'program_uri': program_uri})

    add_to_graph((record_uri, RDF.type, MeMAD.Record))
    add_to_graph((program_uri, MeMAD.hasRecord, record_uri))
    add_to_graph((record_uri, EBUCore.hasIdentifier, Literal(program_id)))
    add_to_graph((record_uri, EBUCore.dateCreated, t_creation_date))
    add_to_graph((record_uri, EBUCore.dateModified, t_update_date))
    add_to_graph((record_uri, EBUCore.hasLanguage, Literal(record_language)))
    add_to_graph((record_uri, EBUCore.hasType,  Literal(record_type)))

    # media
    if program_id.count('_') == 1:
        material_id    = entry['Identifiant Matériels'] if entry['Identifiant Matériels'] else entry['Identifiant Matériels (info.)']
        material_id    = material_id.strip().replace('\r', '')
        material_note  = entry['Matériels  (Détail)'].strip().replace('\r', '')

        media_uri      = encode_uri('media', {'id': program_id})

        add_to_graph((media_uri, RDF.type, EBUCore.MediaResource))
        add_to_graph((program_uri, EBUCore.isInstantiatedBy, media_uri))
        add_to_graph((media_uri, SKOS.note, Literal('Identifiant Matériels: ' + material_id if material_id else None)))
        add_to_graph((media_uri, SKOS.note, Literal('Matériels  (Détail): ' + material_note if material_note else None)))


    # Producers
    producers = entry['Producteurs (Aff.)'].strip().replace('\r', '').split('\n')
    for producer in producers:
        producer = producer.strip()
        if producer:
            add_to_graph((program_uri, EBUCore.hasProducer, Literal(producer)))

    # Contributors
    credits = entry['Générique (Aff. Lig.) '].strip().split(';')
    for credit in credits:
        credit = credit.strip()
        if  credit :
            role, name = credit[:3].strip(), credit[3:].strip()

            if '(' in name:
                name, complement = name.strip('(')[0], name.strip('(')[1:]
                name = name + ' ' + complement[0]

            if '-' in name:
                name, complement = name.strip('-')[0], name.strip('-')[1:]
                name = name + ' ' + complement[0]

            agent_uri = encode_uri('agent', {'name': name})

            add_to_graph((program_uri, EBUCore.hasContributor, agent_uri))
            add_to_graph((agent_uri, RDF.type, EBUCore.Agent))
            add_to_graph((agent_uri, EBUCore.agentName, Literal(name)))
            add_to_graph((agent_uri, EBUCore.hasRole, Literal(role)))


    # Keywords
    keywords = entry['Descripteurs (Aff. Lig.)'].strip().split(';')
    for keyword in keywords:
        keyword = keyword.strip()
        if keyword: 
            keyword = keyword[4:].strip()
            add_to_graph((program_uri, EBUCore.hasKeyword, Literal(keyword)))


    # Genres
    genres = entry['Genre'].strip().split(';')
    for genre in genres:
        genre = genre.strip()
        if genre : 
            add_to_graph((program_uri, EBUCore.hasGenre, Literal(genre)))


    # Themes
    themes     = entry['Thématique'].strip().split(';')
    for theme in themes:
        theme = theme.strip()
        if theme : 
            add_to_graph((program_uri, EBUCore.hasTheme, Literal(theme)))


    # Publication Events
    broadcast_date = entry['Date de diffusion']
    geo_scope = entry['Extension géographique (info.)']

    broadcast_time = entry['Heure de diffusion 2']

    if program_id.count('_') == 2:
        source_program = df_all[df_all['Identifiant de la notice'] == program_id[:-4]]

        try:
            source_program_start = source_program['Heure de diffusion 2'].iloc[0]
            start = time_between(source_program_start, broadcast_time)
            t_start = transform('time', start)
            end = time_after(start, duration[:8])
            t_end = transform('time', end)

            add_to_graph((program_uri, EBUCore.start, t_start))
            add_to_graph((program_uri, EBUCore.end, t_end))
        except Exception as e:
            pass
            
    else:
        t_broadcast_date = transform('datetime', broadcast_date+broadcast_time)
        history_uri  = encode_uri('history', {'program_uri': program_uri})
        pubevent_uri = encode_uri('publication', {'program_uri': program_uri, 'n': '0'})

        add_to_graph((history_uri, RDF.type, EBUCore.PublicationHistory))
        add_to_graph((program_uri, EBUCore.hasPublicationHistory, history_uri))
        add_to_graph((history_uri,  EBUCore.hasPublicationEvent, pubevent_uri))

        add_to_graph((pubevent_uri, RDF.type, EBUCore.PublicationEvent))
        add_to_graph((pubevent_uri, RDF.type, MeMAD.FirstRun))
        add_to_graph((pubevent_uri, EBUCore.publicationStartDateTime, t_broadcast_date))
        add_to_graph((pubevent_uri, EBUCore.publishes, program_uri))
        add_to_graph((pubevent_uri, EBUCore.isReleasedBy, channel_uri))
        add_to_graph((pubevent_uri, EBUCore.duration, transform('duration', duration)))
        add_to_graph((pubevent_uri, EBUCore.hasPublicationRegion, Literal(geo_scope)))


print('Serializing the graph ..')
tick = time.time()
save_graph()
print('Time to save:', round(time.time() - tick, 2), "seconds")

mapping_df = pd.DataFrame(mapping, columns=['identifier', 'URI'])
mapping_df.to_csv('ina_pa_mapping.csv', index=False)




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

	save_graph(path=output_path+'pa_flow_filenames.ttl')
	mapping_all_df = pd.DataFrame(mapping_all)
	mapping_all_df.to_csv('ina_pa_flow_mapping.csv')

	print('INA PA Flow mappings have been succesfully generated.')




if subtitles_path:
    print('Extracting subtitles..')
    d = []

    for filename in tqdm(os.listdir(subtitles_path), total=len(os.listdir(subtitles_path))):
        root = ET.parse(os.path.join(subtitles_path, filename)).getroot()

        speakers = {}
        for i, speaker in enumerate(root.findall('./SpeakerList/Speaker')):
            s = speaker.attrib
            if s['spkid'] in speakers:
                 raise Exception('Speaker already mentioned')
            speakers[s['spkid']] = {'gender':s['gender'], 'lang':s['lang'], 'nw': s['nw']}

        for i, segment in enumerate(root.findall('./SegmentList/SpeechSegment')):
            s = segment.attrib
            content = ' '.join(w.text.strip() for w in segment.findall('./Word'))
            if content == '': continue
            
            entry = {}
            entry['identifier'] = filename
            entry['language'] = s['lang']
            entry['speaker'] = s['spkid']
            entry['gender'] = 'M' if speakers[s['spkid']]['gender'] == '1' else 'F'
            entry['start'] = seconds_to_time(float(s['stime']))
            entry['end'] = seconds_to_time(float(s['etime']))
            entry['content'] = preprocess_content(content)

            d.append(entry)

    df = pd.DataFrame(d)
    df.to_csv('ina_subtitles.csv', index=False)

    print('Mapping subtitles into their program..')

    try:
    	mapping_df = pd.read_csv('ina_ld_mapping.csv').fillna('')
    except:
    	print("Maaping file does not contain some program identifier")

    mapping = {}
    for iden in df.identifier.unique():
        iden = iden[1:] if iden.startswith('R') else iden
        iden = iden.split('.')[0]
        if iden in mapping_df['identifier'].values:
            mapping[iden] = mapping_df[mapping_df['identifier'] == iden].iloc[0]['URI']


    reset_graph()

    counters = {key: 1 for key in mapping}

    for i, entry in tqdm(df.iterrows(), total=len(df)):
        identifier = entry['identifier'][:-4]
        try:
            program_uri  = URIRef(mapping[identifier])
            textline_uri = URIRef(mapping[identifier] + '/subtitles/asr_' + str(counters[identifier]))
            counters[identifier] += 1

            add_to_graph((textline_uri, RDF.type, EBUCore.TextLine))
            add_to_graph((textline_uri, EBUCore.textLineContent, Literal(entry['content'])))
            add_to_graph((textline_uri, EBUCore.textLineLanguage, Literal("FR")))
            add_to_graph((textline_uri, EBUCore.textLineSource, Literal('ASR (Vocapia Research 5.1)')))
            add_to_graph((textline_uri, EBUCore.textLineStartTime, Literal(entry['start'], datatype=XSD.time)))
            add_to_graph((textline_uri, EBUCore.textLineEndTime, Literal(entry['end'], datatype=XSD.time)))
            add_to_graph((textline_uri, EBUCore.hasTextLineRelatedPerson, Literal(entry['speaker'] + entry['gender'])))
            add_to_graph((program_uri, EBUCore.hasRelatedTextLine, textline_uri))
        except:
            print("can't find "+identifier+" in INA PA")

    print('Serializing the subtitles ..')
    tick = time.time()
    save_graph(path=output_path+'ina_subtitles.ttl')
    print('Time to save:', round(time.time() - tick, 2), "seconds")


if not keep_mappings:
	print('Deleting mappings..', end="")
	os.remove("ina_pa_flow_mapping.csv")
	os.remove("ina_pa_mapping.csv")
	os.remove("ina_subtitles.csv")
	print(" Done.")

