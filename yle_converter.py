# -*- coding: utf-8 -*-

import os
import re
import time
import json
import urllib
import pickle
import argparse
import datetime
import unicodedata
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from tqdm import tqdm
from hashlib import sha1
from os.path import exists, dirname, join, isfile
from rdflib import Namespace, URIRef, ConjunctiveGraph, Literal
from rdflib.namespace import FOAF, DC, SKOS, RDF, RDFS, XSD, DCTERMS


parser = argparse.ArgumentParser(description='MeMAD Converter')
parser.add_argument("-p", "--path", type=str, help="Specify the path for the file or folder to process", default='data/yle/') #, required=True)
parser.add_argument("-o", "--output", type=str, help="Specify the path to which the TTL output would be written.", default='data/dump/') #, required=True)
parser.add_argument("-f", "--flow_mapping", type=str, help="Specify the path to a file containing the mapping between filenames and their Flow identifier.", default='data/file_flow_mapping.json') #, required=True)
parser.add_argument("-k", "--keep_mappings", help="add this flag to generate CSV files for mapping Programmes to their URIs", action='store_true', default=True) #, required=True)

# parser.add_argument("-m", "--mode", choices=['metadata', 'flow', 'yle'], type=str, help="Specify which converter to use.", required=True)

args = parser.parse_args()
data_path         = args.path
output_path       = args.output
keep_mappings     = args.keep_mappings
flow_mapping_file = args.flow_mapping

if not exists(data_path) :
	print('Error: the provided path does not exist.')
	exit()

if not exists(output_path):
    print('Creating directory :' + dirname(output_path))
    os.makedirs(dirname(output_path))

data_path        = data_path + '/' if data_path[-1] != '/' else data_path
output_path      = output_path + '/' if output_path[-1] != '/' else output_path
repos_to_process = sorted(os.listdir(data_path))

# if given a path that directly contains metadata files 
if repos_to_process[0][-3:] == 'xml':
	dataset_name = data_path.split('/')[-2]
	data_path = data_path[:-(len(dataset_name)+1)]
	repos_to_process = [dataset_name]
	print('Processing the "', dataset_name, '" dataset @', data_path)
else:
	print('Processing', len(repos_to_process), 'datasets..')



MeMAD   = Namespace('http://data.memad.eu/ontology#')
EBUCore = Namespace('http://www.ebu.ch/metadata/ontologies/ebucore/ebucore#')

base = 'http://data.memad.eu/'

def save_graph(path='yle.ttl'):
    g.serialize(path, format='turtle')

def reset_graph(path=''):
    global g
    g = ConjunctiveGraph()
    if path:
        g.load(path, format='turtle')
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
    to_dash = '\\/\',.":;[]()!? #=&$%@{«°»¿=>+*'
    cleaned = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    cleaned = ''.join('-' if c in to_dash else c for c in cleaned)
    cleaned = ''.join(c if i == 0 or (c == '-' and cleaned[i-1]) != '-' else '' for i, c in enumerate(cleaned))
    cleaned = cleaned.lower().strip('-')
    return cleaned
    
def transform(field, value):
    """ Transform some values to a proper format : e.g. duration, URNs .. """ 
    """ Returns a rdflib.URI or a rdflib.Literal """
    if not value: 
        return None

    elif field == 'aspect_ratio':
        aspect_ratios = json.load(open('mappings/yle_aspect_ratio.json'))
        return Literal(aspect_ratios[value])

    elif field == 'episode_language':
        languages = json.load(open('mappings/yle_episode_lang.json'))
        # return Literal(languages[value.lower()])
        return languages[value.lower()]

    elif field == 'duration_tc':
        h, m, s, ms = value.split(':')
        value = 'PT'+h+'H'+m+'M'+s + ('.'+ms if ms != '00' else '') +'S'
        return Literal(value, datatype=XSD.duration)

    elif field == 'time':
        ms = int(value)
        s = int((ms/1000)%60)
        s = str(s).zfill(2)
        m = int((ms/(1000*60))%60)
        m = str(m).zfill(2)
        h = int((ms/(1000*60*60))%24)
        h = str(h).zfill(2)
        ms = str(int(ms % 1000)).zfill(3)

        return Literal(h + ':' + m + ':' + s + '.' + ms, datatype=XSD.time)

    elif field == 'video_format':
        EBU_VideoCompression_schema = Namespace('http://www.ebu.ch/metadata/ontologies/skos/ebu_VideoCompressionCodeCS#')
        video_formats= {'0': EBU_VideoCompression_schema['_12'], # PAL 
                        '1': EBU_VideoCompression_schema['_12'], # PAL
                        '2': EBU_VideoCompression_schema['_14'], # NTSC
                        '3': EBU_VideoCompression_schema['_15']} # SECAM
        return video_formats[value]

    elif field == 'date':
        Y, M, D = value[:4], value[4:6], value[6:8]
        value = '{}-{}-{}'.format(Y, M, D)
        return Literal(value, datatype=XSD.date)

    elif field == 'datetime':
        Y, M, D = value[:4], value[4:6], value[6:8]
        h, m, s = value[8:10], value[10:12], value[12:14]
        value = '{}-{}-{}T{}:{}:{}'.format(Y, M, D, h, m, s)  # 1972-12-31T00:00:00
        return Literal(value, datatype=XSD.dateTime)

    elif field == 'subtitles_language':
        subtitles_languages = json.load(open('mappings/yle_subtitles_lang.json'))
        return subtitles_languages[value.lower()]

    elif field == 'sub_format':
        formats = {}
        return Literal(formats[value])

    elif field == 'audio_language':
        audio_languages = json.load(open('mappings/yle_audio_lang.json'))
        return audio_languages[value.lower()]

    elif field == 'contributor_role':
        roles = json.load(open('mappings/yle_id2role_en.json'))
        return roles[value] # value is the id

    else:
        raise Exception('Field ' + field + ' isn\'t mapped for value ' + value)


def encode_uri(resource, data):
    if resource == 'program':
        hashed = sha1(data['id'].encode()).hexdigest()
        source = data['source']
        parent = clean_string(data['parent'])
        return URIRef(base + source + '/' + parent + '/' + hashed)
    elif resource == 'collection':
        name_clean = clean_string(data['name'])
        source = data['source']
        return URIRef(base + source + '/' + name_clean)
    elif resource == 'publication':
        n = str(data['n'])
        return URIRef(str(data['program_uri']) + '/publication/' + n)
    elif resource == 'history':
        return URIRef(str(data['program_uri']) + '/publication')
    elif resource == 'media':
        hashed = sha1(data['id'].encode()).hexdigest() 
        return URIRef(base + 'media/' + hashed)
    elif resource == 'subtitling':
        return URIRef(str(data['program_uri'] + '/subtitling/' + str(data['n'])))
    elif resource == 'audio':
        return URIRef(str(data['program_uri'] + '/audio/' + str(data['n'])))
    elif resource == 'channel':
        channel_codes = json.load(open('mappings/yle_channel2code.json'))
        return URIRef(base + 'channel/' + channel_codes[data['name']])
    elif resource == 'agent':
        agent_name_clean = clean_string(data['name'])
        return URIRef(base + 'agent/' + agent_name_clean)

    elif resource == 'language':
        language = str(data['language']).lower().replace(' ', '_')
        return URIRef(base + 'language/' + language)
    elif resource == 'role':
        role = str(data['role']).lower().replace(' ', '_')
        return URIRef(base + 'role/' + role)

    elif resource == 'genre':
        genres = json.load(open('mappings/yle_class2label.json'))

        genre_fi = data['genre']
        if genre_fi in genres:
            genre_en = genres[genre_fi]
            return URIRef(base + 'genre/' + genre_en.lower().replace(' ', '_').replace('/', '_'))
        else:
            return Literal(genre_fi, lang='fi')

    else:
        raise Exception('No URI encoding for resource ' + resource)


def add_languages():
    mapping_files = ['yle_episode_lang.json', 'yle_subtitles_lang.json', 'yle_audio_lang.json']
    unique_values = set()

    for file in mapping_files:
        dic = json.load(open('mappings/' + file))
        for v in dic.values():
            for l in v.split('/'):
                unique_values.add(l.lower())

    # print('Adding the following languages to the graph:', ', '.join(sorted(unique_values)))

    for language in unique_values:
        language_uri = URIRef(base + 'language/' + language.lower().replace(' ', '_'))
        langauge_label = language[0].upper() + language[1:]
        add_to_graph((language_uri, RDF.type, EBUCore.Language))
        add_to_graph((language_uri, RDFS.label, Literal(langauge_label)))

def add_roles():
    roles_fi = json.load(open('mappings/yle_id2role.json'))
    roles_en = json.load(open('mappings/yle_id2role_en.json'))

    # print('Adding the following roles to the graph:', ', '.join(sorted(roles_en.values())))
    
    for code, label_fi in roles_fi.items():
        label_en = roles_en[code]
        role_uri = URIRef(base + 'role/' + label_en.lower().replace(' ', '_').replace('/', '_'))
        add_to_graph((role_uri, RDF.type, EBUCore.Role))
        add_to_graph((role_uri, RDFS.label, Literal(label_en)))
        add_to_graph((role_uri, RDFS.label, Literal(label_fi, lang='fi')))


def add_genres():
    genres = json.load(open('mappings/yle_class2label.json'))

    for genre_fi, genre_en in genres.items():
        genre_uri = URIRef(base + 'genre/' + genre_en.lower().replace(' ', '_').replace('/', '_'))
        genre_label = genre_en[0].upper() + genre_en[1:]
        add_to_graph((genre_uri, RDF.type, EBUCore.Genre))
        add_to_graph((genre_uri, RDFS.label, Literal(genre_label)))
        add_to_graph((genre_uri, RDFS.label, Literal(genre_fi, lang='fi')))


mapping = []
segments_mapping = []
class_subs = set()

tick = time.time()
for dataset in repos_to_process: # ['14-may2019']: # 
    if '.' in dataset:
        continue
    print('Processing', dataset, '..')
    reset_graph()
    
    # Adding controlled vocabularies to the Knowledge Graph
    add_languages()
    add_roles()
    add_genres()
    
    files = os.listdir(data_path+dataset)
    for ii, file in enumerate(tqdm(files)):

        root = ET.parse(data_path+dataset+'/'+file).getroot()
        guid = root.find("./MAObject[1]/GUID").text

        # Source
        series_id      = root.find("./MAObject[1]/Meta/[@name='SERIES_ID']").text
        series_name    = root.find("./MAObject[1]/Meta/[@name='SERIES_NAME']").text
        
        parent = 'orphan'
        if series_name:
            series_uri = encode_uri('collection', {'name': series_name, 'source': 'yle'})
            parent = series_name
        
        program_uri = encode_uri('program', {'id': guid, 'parent': parent, 'source': 'yle'})
        
        if series_name:
            add_to_graph((series_uri, RDF.type, EBUCore.Series))
            add_to_graph((series_uri, RDF.type, EBUCore.Collection))
            add_to_graph((series_uri, EBUCore.title, Literal(series_name)))
            add_to_graph((series_uri, EBUCore.hasIdentifier, Literal(series_id)))
            add_to_graph((series_uri, EBUCore.isParentOf, program_uri))
            add_to_graph((program_uri, RDF.type, EBUCore.Episode))

        
        # Metadata
        dataset        = dataset
        filename       = file
        number         = root.find("./MAObject[1]/Meta/[@name='EPISODE_NUMBER']").text
        subject        = root.find("./MAObject[1]/Meta/[@name='SUBJECT']").text
        fi_title       = root.find("./MAObject[1]/Meta/[@name='FI_TITLE']").text
        se_title       = root.find("./MAObject[1]/Meta/[@name='SE_TITLE']").text
        version        = root.find("./MAObject[1]/Meta/[@name='VERSION_NAME']").text 
        main_title     = root.find("./MAObject[1]/Meta/[@name='MAINTITLE']").text
        metro_id       = root.find("./MAObject[1]/Meta/[@name='METRO_PROGRAMME_ID']").text
        language       = root.find("./MAObject[1]/Meta/[@name='LANGUAGE']").text
        duration1      = root.find("./MAObject[1]/Meta/[@name='DURATION']").text
        description    = root.find("./MAObject[1]/Meta/[@name='DESCRIPTION_SHORT']").text
        duration_tc    = root.find("./MAObject[1]/Meta/[@name='SYSTEM_DURATION_TC']").text 
        working_title  = root.find("./MAObject[1]/Meta/[@name='WORKING_TITLE']").text
        archiving_date = root.find("./MAObject[1]/Meta/[@name='ARCHIVE_DATE']").text 
        class_comb_a   = root.find("./MAObject[1]/Meta/[@name='CLASSIFICATION_COMB_A']").text 
        class_content  = root.find("./MAObject[1]/Meta/[@name='CLASSIFICATION_CONTENT']").text 
        class_main     = root.find("./MAObject[1]/Meta/[@name='CLASSIFICATION_MAIN_CLASS']").text 
        class_sub      = root.find("./MAObject[1]/Meta/[@name='CLASSIFICATION_SUB_CLASS']").text         
        web_desc       = root.find("./MAObject[1]/Meta/[@name='WEB_DESCRIPTION']").text 
        web_desc_sw    = root.find("./MAObject[1]/Meta/[@name='WEB_DESCRIPTION_SWE']").text 

        class_comb_a   = root.find("./MAObject[1]/Meta/[@name='CLASSIFICATION_COMB_A']").text 
        class_content  = root.find("./MAObject[1]/Meta/[@name='CLASSIFICATION_CONTENT']").text 
        class_main     = root.find("./MAObject[1]/Meta/[@name='CLASSIFICATION_MAIN_CLASS']").text 
        class_sub      = root.find("./MAObject[1]/Meta/[@name='CLASSIFICATION_SUB_CLASS']").text         

        languages      = transform('episode_language', language)
        duration_tc    = transform('duration_tc', duration_tc)
        archiving_date = transform('date', archiving_date)
        class_sub      = class_sub if ']' not in class_sub else class_sub.split(']')[1][1:]
        class_subs.add(class_sub)

        class_comb_a_uri   = encode_uri('genre', {'genre': class_comb_a})
        class_content_uri  = encode_uri('genre', {'genre': class_content})
        class_main_uri     = encode_uri('genre', {'genre': class_main})
        class_sub_uri      = encode_uri('genre', {'genre': class_sub})


        add_to_graph((program_uri, RDF.type, EBUCore.TVProgramme))
        add_to_graph((program_uri, DCTERMS.publisher, Literal("Yle")))
        add_to_graph((program_uri, EBUCore.hasIdentifier, Literal(guid)))
        add_to_graph((program_uri, EBUCore.hasIdentifier, Literal(filename.split('.')[0])))
        # add_to_graph((program_uri, EBUCore.filename, Literal(filename)))
        add_to_graph((program_uri, EBUCore.hasSubject, Literal(subject)))
        add_to_graph((program_uri, EBUCore.episodeNumber, Literal(number)))
        add_to_graph((program_uri, EBUCore.description, Literal(description)))
        add_to_graph((program_uri, EBUCore.title, Literal(fi_title)))
        add_to_graph((program_uri, EBUCore.title, Literal(se_title, lang='se')))
        add_to_graph((program_uri, EBUCore.mainTitle, Literal(main_title)))
        add_to_graph((program_uri, EBUCore.hasLanguage, Literal(languages)))
        add_to_graph((program_uri, EBUCore.duration, duration_tc))
        add_to_graph((program_uri, EBUCore.version, Literal(version)))
        add_to_graph((program_uri, EBUCore.workingTitle, Literal(working_title)))
        add_to_graph((program_uri, EBUCore.dateArchived, archiving_date))
        add_to_graph((program_uri, EBUCore.description, Literal(web_desc)))
        add_to_graph((program_uri, EBUCore.description, Literal(web_desc_sw, lang='se')))
        add_to_graph((program_uri, EBUCore.hasGenre, class_content_uri))
        add_to_graph((program_uri, EBUCore.hasGenre, class_comb_a_uri))
        add_to_graph((program_uri, EBUCore.hasGenre, class_main_uri))
        add_to_graph((program_uri, EBUCore.hasGenre, class_sub_uri))
        
        if languages is not None:
            for language in languages.split('/'):
                language_uri = encode_uri('language', {'language': language})
                add_to_graph((program_uri, EBUCore.hasLanguage, language_uri))
        
        # Media
        media_id           = root.find("./MAObject[1]/Meta/[@name='MEDIA_ID']").text
        media_framerate    = root.find("./MAObject[1]/Meta/[@name='SYSTEM_FRAMERATE_FPS']").text 
        media_video_format = root.find("./MAObject[1]/Meta/[@name='VIDEO_FORMAT']").text 
        media_aspect_ratio = root.find("./MAObject[1]/Meta/[@name='ASPECT_RATIO']").text
        metro_id           = root.find("./MAObject[1]/Meta/[@name='METRO_PROGRAMME_ID']").text

        media_uri     = encode_uri('media', {'id': guid})
        
        media_aspect_ratio_uri = transform('aspect_ratio', media_aspect_ratio)
        media_video_format     = transform('video_format', media_video_format)
        
        add_to_graph((media_uri, RDF.type, EBUCore.MediaResource))   
        add_to_graph((program_uri, EBUCore.isInstantiatedBy, media_uri))
        # add_to_graph((program_uri, EBUCore.filename, Literal(file)))
        add_to_graph((media_uri, MeMAD.hasMetroIdentifier, Literal(metro_id)))
        add_to_graph((media_uri, EBUCore.aspectRatio, media_aspect_ratio_uri))
        add_to_graph((media_uri, EBUCore.hasVideoEncodingFormat, media_video_format))
        add_to_graph((media_uri, EBUCore.frameRate, Literal(media_framerate, datatype=XSD.float)))

        # subtitles
        subtitles    = root.findall("./MVAttribute[@type='SUBTITLES']")
        for i, subtitle in enumerate(subtitles):
            subtitles_filename      = subtitle.find("./Meta[@name='ST_FILENAME']").text
            subtitles_language      = subtitle.find("./Meta[@name='ST_LANGUAGE_CODES']").text
            subtitles_file_format   = subtitle.find("./Meta[@name='ST_FILE_FORMAT']").text
            subtitles_date_ingested = subtitle.find("./Meta[@name='ST_INGEST_DATE']").text
            subtitles_date_publised = subtitle.find("./Meta[@name='ST_PUB_DATE']").text

            subtitles_language      = transform('subtitles_language', subtitles_language)
            subtitles_file_format   = transform('sub_format', subtitles_file_format)
            subtitles_date_ingested = transform('date', subtitles_date_ingested)
            subtitles_date_publised = transform('date', subtitles_date_publised)

            subtitles_uri = encode_uri('subtitling', {'n':i, 'program_uri': program_uri})
            subtitles_language_uri = encode_uri('language', {'language': subtitles_language})

            add_to_graph((subtitles_uri, RDF.type, EBUCore.Subtitling))
            add_to_graph((program_uri, EBUCore.hasSubtitling, subtitles_uri))
            add_to_graph((subtitles_uri, EBUCore.hasLanguage, subtitles_language_uri))
            add_to_graph((subtitles_uri, EBUCore.filename, Literal(subtitles_filename)))
            add_to_graph((subtitles_uri, EBUCore.hasFileFormat, subtitles_file_format))
            add_to_graph((subtitles_uri, EBUCore.dateIngested, subtitles_date_ingested))
            add_to_graph((subtitles_uri, EBUCore.datePublished, subtitles_date_publised))
        
        # Audio
        audios = root.findall("./MVAttribute[@type='AUDIO']")
        for i, audio in enumerate(audios):
            audio_codecs       = audio.find("./Meta[@name='PMA_CODEC']").text
            audio_language     = audio.find("./Meta[@name='PMA_LANGUAGE']").text
            audio_sample_rates = audio.find("./Meta[@name='PMA_SAMPLE_RATE']").text
            audio_note         = audio.find("./Meta[@name='PMA_NOTES']").text

            audio_language = transform('audio_language', audio_language)
            audio_language_uri = encode_uri('language', {'language': audio_language})

            audio_uri = encode_uri('audio', {'n':i, 'program_uri': program_uri})

            add_to_graph((audio_uri, RDF.type, EBUCore.AudioTrack))
            add_to_graph((program_uri, EBUCore.hasAudioTrack, audio_uri))
            add_to_graph((audio_uri, EBUCore.hasLanguage, audio_language_uri))
            add_to_graph((audio_uri, SKOS.note, Literal(audio_note)))
            add_to_graph((audio_uri, EBUCore.sampleRate, Literal(audio_sample_rates, datatype=XSD.nonNegativeInteger)))

        
        # Pubevents
        history_uri  = encode_uri('history', {'program_uri': program_uri})
        
        add_to_graph((history_uri, RDF.type, EBUCore.PublicationHistory))
        add_to_graph((program_uri, EBUCore.hasPublicationHistory, history_uri))
        
        firstrun_date = root.find("./MAObject[1]/Meta/[@name='FIRSTRUN_DATE']").text
        firstrun_time = root.find("./MAObject[1]/Meta/[@name='FIRSTRUN_TIME']").text
        if firstrun_date and firstrun_time:
            first_pub_uri = encode_uri('publication', {'program_uri': program_uri, 'n': 'firstrun'})
            firstrun_datetime = transform('datetime', firstrun_date + firstrun_time)

            add_to_graph((history_uri,  EBUCore.hasPublicationEvent, first_pub_uri))
            add_to_graph((first_pub_uri, RDF.type, MeMAD.FirstRun))
            add_to_graph((first_pub_uri, EBUCore.publicationStartDateTime, firstrun_datetime))
            add_to_graph((first_pub_uri, EBUCore.publishes, program_uri))
        
        pubevents    = root.findall("./MVAttribute[@type='PUBLICATIONS']")
        for i, pubevent in enumerate(pubevents):
            pubevent_id           = pubevent.find("./Meta[@name='PUB_ID']").text
            pubevent_datetime     = pubevent.find("./Meta[@name='PUB_DATETIME']").text
            pubevent_datetime_end = pubevent.find("./Meta[@name='PUB_DATETIME_END']").text

            pubevent_datetime     = transform('datetime', pubevent_datetime)
            pubevent_datetime_end = transform('datetime', pubevent_datetime_end)
            
            channel_name = pubevent.find("./Meta[@name='PUB_CHANNEL']").text            
            channel_uri  = encode_uri('channel', {'name': channel_name})
            channel_code = str(channel_uri).split('/')[-1]

            add_to_graph((channel_uri, RDF.type, EBUCore.PublicationChannel))
            add_to_graph((channel_uri, EBUCore.publicationChannelName, Literal(channel_name)))
            add_to_graph((channel_uri, EBUCore.publicationChannelId, Literal(channel_code)))
            add_to_graph((channel_uri, EBUCore.serviceDescription, Literal('TV channel')))
            
            pubevent_uri = encode_uri('publication', {'program_uri': program_uri, 'datetime':pubevent_datetime, 'n': i})
            add_to_graph((pubevent_uri, RDF.type, EBUCore.PublicationEvent))
            add_to_graph((history_uri,  EBUCore.hasPublicationEvent, pubevent_uri))
            add_to_graph((pubevent_uri, EBUCore.publishes, program_uri))
            add_to_graph((pubevent_uri, EBUCore.isReleasedBy, channel_uri))
            add_to_graph((pubevent_uri, EBUCore.publicationStartDateTime, pubevent_datetime))
            add_to_graph((pubevent_uri, EBUCore.publicationEndDateTime, pubevent_datetime_end))
    
            if i == 0 and not (firstrun_date and firstrun_time):
                add_to_graph((pubevent_uri, RDF.type, MeMAD.FirstRun))

            if i == 0:
                add_to_graph((pubevent_uri, EBUCore.firstShowing, Literal("1", datatype=XSD.boolean)))
            
    
        # Content Segments
        segments_content = {}
        contents     = root.findall("./MAObject[@mdclass='S_CONTENT_DESCRIPTION']")
        for content in contents:
            segment_guid = content.find('./GUID').text
            segment_description = content.find('./Meta[@name="SEGMENT_DESCRIPTION"]').text
            segments_content[segment_guid] = segment_description
        
        # next we get the timing metadata for each segment from the corresponding stratum
        # and map them to their description via GUID
        segments     = root.findall("./MAObject[1]/StratumEx[@name='CONTENT_DESCRIPTION']/*/Segment")
        for i, segment in enumerate(segments) :
            if 'contentid' in segment.attrib: # if the media segment has a description
                segment_content_id  = segment.attrib['contentid']
                
                if segment_content_id in segments_content:
                    segment_begin       = segment.attrib['begin']
                    segment_end         = segment.attrib['end']
                    segment_description = segments_content[segment_content_id]

                    segment_uri = encode_uri('program', {'id': segment_content_id, 'parent': parent, 'source': 'yle'})

                    # Check whether this actually works
                    segment_dur   = transform('time', str(int(segment_end) - int(segment_begin)))
                    segment_start = transform('time', segment_begin)
                    segment_end   = transform('time', segment_end)

                    add_to_graph((segment_uri, RDF.type, EBUCore.Part))
                    add_to_graph((program_uri, EBUCore.hasPart, segment_uri))
                    add_to_graph((segment_uri, EBUCore.start, segment_start))
                    add_to_graph((segment_uri, EBUCore.end, segment_end))
                    add_to_graph((segment_uri, EBUCore.duration, segment_dur))
                    add_to_graph((segment_uri, EBUCore.description, Literal(segment_description)))
                    
                    segments_mapping.append((segment_content_id, str(program_uri), file, str(segment_start), str(segment_end)))

        # Contributors
        contributors = root.findall("./MVAttribute[@type='CONTRIBUTORS']")
        for i, contributor in enumerate(contributors):
            contributor_name = contributor.find('./Meta[@name="CONT_PERSON_NAME"]').text
            contributor_role = contributor.find('./Meta[@name="CONT_PERSON_ROLE"]').text

            if contributor_name:
                try:
                    contributor_uri = encode_uri('agent', {'name':contributor_name.strip()})

                    add_to_graph((contributor_uri, RDF.type, EBUCore.Agent))
                    add_to_graph((program_uri, EBUCore.hasContributor, contributor_uri))
                    add_to_graph((contributor_uri, EBUCore.agentName, Literal(contributor_name)))
                    
                    if contributor_role:
                        contributor_role = transform('contributor_role', contributor_role.strip())
                        contributor_role_uri = encode_uri('role', {'role': contributor_role})
                        add_to_graph((contributor_uri, EBUCore.hasRole, contributor_role_uri))
                except:
                    print(filename, contributor_role)

            
        mapping.append((file, str(program_uri)))
    
    print('saving into', f'{output_path}{dataset}.ttl')
    save_graph(path=f'{output_path}{dataset}.ttl')

print('Time elapsed:', round(time.time() - tick, 2), 'seconds.')

mapping_df = pd.DataFrame(mapping, columns=['identifier', 'URI'])
mapping_df.to_csv('yle_mapping.csv', index=False)

segments_mapping_df = pd.DataFrame(segments_mapping, columns=['identifier', 'URI', 'filename', 'start', 'end'])
segments_mapping_df.to_csv('yle_segments_mapping.csv', index=False)


if flow_mapping_file:
	print('FLOW triplets generation..')

	data = json.load(open(flow_mapping_file, 'r'))

	yle_filenames = []
	for dataset in os.listdir(data_path): # ['14-may2019']: # 
	    if '.' in dataset: continue

	    files = os.listdir(data_path+dataset)
	    for file in files:
	        yle_filenames.append(file)

	extentions = set()
	identifiers = []

	reset_graph()
	found = []

	mapping_all = []

	for obj in data:
	    filename = obj['name']
	    if '.' not in filename: continue # "Political Debates"

	    flow_href  = URIRef(obj['flowHRef'])
	    identifier = filename.split('.')[0]
	    extention  = filename.split('.')[1]

	    identifiers.append(identifier)
	    extentions.add(extention)

	    try:
	        program     = mapping_df[mapping_df['identifier'] == identifier + '.xml'].iloc[0]
	    except:
	        try:
	            program     = mapping_df[mapping_df['identifier'] == identifier.replace('MEDIA', 'PROG')  + '.xml'].iloc[0]
	        except:
	            continue
	    
	    if identifier + '.xml' in yle_filenames:
	        media_uri = URIRef(base + 'media/' + str(program['URI']).split('/')[-1])
	        
	        add_to_graph((media_uri, EBUCore.locator, flow_href))
	        add_to_graph((media_uri, EBUCore.filename, Literal(filename)))
	        
	        
	        mapping_all.append({'uri':str(media_uri), 'flow_href': str(flow_href), 'filename': filename})
	        found.append(identifier + '.xml')
	        
	    if identifier.replace('MEDIA', 'PROG') + '.xml' in yle_filenames:
	        found.append(identifier + '.xml')


	save_graph(path=output_path+'yle_flow_filenames.ttl')


	# print('Found extentions: ')
	# print(extentions)


	mapping_all_df = pd.DataFrame(mapping_all)
	mapping_all_df.to_csv('yle_filename_flow_mapping.csv', index=False)

	print('Yle Flow mappings have been succesfully generated.')

if not keep_mappings:
	print('Deleting mappings.. Done.')
	os.remove("yle_filename_flow_mapping.csv")
	os.remove("yle_mapping.csv")
	os.remove("yle_segments_mapping.csv")
