# -*- coding: utf-8 -*-

import time
import unicodedata
import json
import os
import rdflib
from os.path import exists, dirname
from hashlib import sha1
from rdflib import Namespace, URIRef, ConjunctiveGraph, Literal
from rdflib.namespace import FOAF, DC, SKOS, RDF, XSD
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


# Declaring all Namespaces
INA     = Namespace('http://www.ina.fr/core#')
Yle     = Namespace('http://www.yle.fi/ontology#')
MeMAD   = Namespace('http://data.memad.eu/ontology#')
EBUCore = Namespace('http://www.ebu.ch/metadata/ontologies/ebucore/ebucore#')

base = 'http://data.memad.eu/'


###########################
#    Yle XML2RDF class    #
###########################


class Yle_Store:
    """ A class for storing and handling Yle datasets """
    """ Constructed from files containing metadata on Episodes, Series, Segment"""

    def __init__(self, store_path = None):
        """ Load a graph from a given path or create a blank one + bind namespaces to prefixes """
        # creating or loading the in-memory graph
        self.graph = ConjunctiveGraph()
        self.graph.bind('memad', MeMAD)
        self.graph.bind('ebucore', EBUCore)
        
        self.path  = store_path
        if self.path and exists(self.path):
            self.graph.load(self.path, format='turtle')
        elif not exists(dirname(self.path)):
            print('Creating directory :' + dirname(self.path))
            os.makedirs(dirname(self.path))

        
    def save(self, save_path = None):
        """ serialize the graph into Turtle (to different path if the argument is given) """
        self.graph.serialize(save_apth if save_path else self.path, format='turtle')

    def process_file(self, file, autosave=False):
        """ add the metadata from the file to the in-memory store """
        root = ET.parse(file).getroot()

        # adding metadata about the program, the series (if found) and the media 
        program_uri, program_id = self.add_episode_metadata(root)
        self.add_media_metadata(root, program_uri)
        
        # Strata metadata: Subtitles, Audio tracks, Publication Events, Contributors, Segments
        self.add_subtitles_metadata(root, program_uri)
        self.add_audio_metadata(root, program_uri)
        self.add_pubevents_metadata(root, program_uri)
        self.add_segments_metadata(root, program_uri)
        self.add_contributors_metadata(root, program_uri)
        
        if autosave:
            self.save()
    

    def add_to_graph(self, triplet):
        if triplet[2] and str(triplet[2]) != 'None': # the predicate has a non-null value
            self.graph.add(triplet)
        else:
            # print(triplet[1], 'was not added to the graph')
            pass
    
    def clean(self, s):
        """ Transforming free text strings into ascii slugs """
        to_dash = '\\/\',.":;[]()!? #=&'
        cleaned = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        cleaned = ''.join('-' if c in to_dash else c for c in cleaned)
        cleaned = ''.join(c if i == 0 or (c == '-' and cleaned[i-1]) != '-' else '' for i,c in enumerate(cleaned))
        cleaned = cleaned.lower().strip('-')
        return cleaned


    def encode_uri(self, resource, data):
        if resource == 'program':
            hashed = sha1(data['guid'].encode()).hexdigest()
            source = data['source']
            parent = self.clean(data['parent'])
            return URIRef(base + source + '/' + parent + '/' + hashed)
        elif resource == 'series':
            series_name_clean = self.clean(data['name'])
            source = data['source']
            return URIRef(base + source + '/' + series_name_clean)
        elif resource == 'publication':
            date = ''.join(c for c in data['datetime'] if c in '0123456789')
            n = str(data['n'])
            return URIRef(str(data['program_uri'] + '/publication/' + n))
        elif resource == 'media':
            hashed = sha1(data['media_id'].encode()).hexdigest() 
            return URIRef(base + 'media/' + hashed)
        elif resource == 'subtitling':
            return URIRef(str(data['program_uri'] + '/subtitling/' + str(data['n'])))
        elif resource == 'audio':
            return URIRef(str(data['program_uri'] + '/audio/' + str(data['n'])))
        elif resource == 'channel':
            channel_codes = json.load(open('mappings/yle_channel2code.json'))
            return URIRef(base + 'channel/' + channel_codes[data['name']])
        elif resource == 'agent':
            agent_name_clean = self.clean(data['name'])
            return URIRef(base + 'agent/' + agent_name_clean)
        elif resource == 'segment':
            hashed = sha1(data['guid'].encode()).hexdigest() 
            base_uri = '/'.join(str(data['program_uri']).split('/')[:-1])
            return URIRef(base_uri + '/' + hashed)
        else:
            raise Exception('Field "' + field + '" is not covered.')
    

    def transform(self, field, value):
        """ Transform some values to a proper format : e.g. duration, URNs .. """ 
        """ Returns a rdflib.URI or a rdflib.Literal """
        if not value: 
            return None
        
        elif field == 'aspect_ratio':
            aspect_ratios = json.load(open('mappings/yle_aspect_ratio.json'))
            return Literal(aspect_ratios[value])
        
        elif field == 'episode_language':
            languages = json.load(open('mappings/yle_episode_lang.json'))
            return Literal(languages[value])
        
        elif field == 'duration_tc':
            h, m, s, ms = value.split(':')
            value = 'PT'+h+'H'+m+'M'+s+'S'
            return Literal(value, datatype=XSD.duration)
        
        elif field == 'time':
            ms = int(value)
            s = int((ms/1000)%60)
            s = str(s).zfill(2)
            m = int((ms/(1000*60))%60)
            m = str(m).zfill(2)
            h = int((ms/(1000*60*60))%24)
            h = str(h).zfill(2)
            ms = str(int(ms % 1000))

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
            Literal(subtitles_languages[value])
            
        elif field == 'sub_format':
            formats = {}
            return Literal(formats[value])

        elif field == 'audio_language':
            audio_languages = json.load(open('mappings/yle_audio_lang.json'))
            return Literal(audio_languages[value])
        
        elif field == 'contributor_role':
            roles = json.load(open('mappings/yle_id2role.json'))
            role = roles[value] # value is the id
            return Literal(role, lang='fi')
        
        else:
            raise Exception('Field ' + field + ' isn\'t mapped for value ' + value)

    
    def add_episode_metadata(self, root):
        # Episode, Series and Media metadata
        guid           = root.find("./MAObject[1]/GUID").text
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
    
        t_language_uri   = self.transform('episode_language', language)
        t_duration_tc    = self.transform('duration_tc', duration_tc)

        series_id      = root.find("./MAObject[1]/Meta/[@name='SERIES_ID']").text
        series_name    = root.find("./MAObject[1]/Meta/[@name='SERIES_NAME']").text
        if series_id and series_name:
            program_uri = self.encode_uri('program', {'guid': guid, 'parent': series_name, 'source': 'yle'})
            self.add_to_graph((program_uri, RDF.type, EBUCore.Episode))
            self.add_series_metadata(root, program_uri)
        
        else:
            program_uri = self.encode_uri('program', {'guid': guid, 'parent': 'orphan', 'source': 'yle'})
            self.add_to_graph((program_uri, RDF.type, EBUCore.TVProgramme))

        firstrun_date = root.find("./MAObject[1]/Meta/[@name='FIRSTRUN_DATE']").text
        firstrun_time = root.find("./MAObject[1]/Meta/[@name='FIRSTRUN_TIME']").text
        if firstrun_date and firstrun_time:
            self.add_firstrun_metadata(root, program_uri)
        
        self.add_to_graph((program_uri, RDF.type, EBUCore.TVProgramme))
        self.add_to_graph((program_uri, EBUCore.hasIdentifier, Literal(guid)))
        self.add_to_graph((program_uri, MeMAD.hasMetroIdentifier, Literal(metro_id)))
        self.add_to_graph((program_uri, EBUCore.episodeNumber, Literal(number, datatype=XSD.nonNegativeInteger)))
        self.add_to_graph((program_uri, MeMAD.hasGUID, Literal(guid)))
        self.add_to_graph((program_uri, EBUCore.description, Literal(description, lang='fi')))
        self.add_to_graph((program_uri, EBUCore.title, Literal(fi_title, lang='fi')))
        self.add_to_graph((program_uri, EBUCore.title, Literal(se_title, lang='se')))
        self.add_to_graph((program_uri, EBUCore.mainTitle, Literal(main_title)))
        self.add_to_graph((program_uri, EBUCore.hasLanguage, t_language_uri))
        self.add_to_graph((program_uri, EBUCore.duration, t_duration_tc))
        self.add_to_graph((program_uri, EBUCore.version, Literal(version)))
        self.add_to_graph((program_uri, EBUCore.workingTitle, Literal(working_title)))
        
        return program_uri, guid

    def add_series_metadata(self, root, program_uri):
        series_id   = root.find("./MAObject[1]/Meta/[@name='SERIES_ID']").text
        series_name = root.find("./MAObject[1]/Meta/[@name='SERIES_NAME']").text
        
        series_uri    = self.encode_uri('series', {'name': series_name, 'source': 'yle'})
        
        self.add_to_graph((series_uri, RDF.type, EBUCore.Series))
        self.add_to_graph((series_uri, EBUCore.title, Literal(series_name)))
        self.add_to_graph((series_uri, EBUCore.hasEpisode, program_uri))
    
    def add_firstrun_metadata(self, root, program_uri):
        firstrun_date = root.find("./MAObject[1]/Meta/[@name='FIRSTRUN_DATE']").text
        firstrun_time = root.find("./MAObject[1]/Meta/[@name='FIRSTRUN_TIME']").text

        first_pub_uri = self.encode_uri('publication', {'program_uri': program_uri, 'datetime': firstrun_date+firstrun_time, 'n': '0'})
        
        t_firstrun_datetime = self.transform('datetime', firstrun_date + firstrun_time)
        
        self.add_to_graph((first_pub_uri, EBUCore.publishes, program_uri))
        self.add_to_graph((first_pub_uri, RDF.type, MeMAD.FirstRun))
        self.add_to_graph((first_pub_uri, EBUCore.publicationStartDateTime, t_firstrun_datetime))
        
    def add_media_metadata(self, root, program_uri):
        media_id           = root.find("./MAObject[1]/Meta/[@name='MEDIA_ID']").text
        media_framerate    = root.find("./MAObject[1]/Meta/[@name='SYSTEM_FRAMERATE_FPS']").text 
        media_video_format = root.find("./MAObject[1]/Meta/[@name='VIDEO_FORMAT']").text 
        media_aspect_ratio = root.find("./MAObject[1]/Meta/[@name='ASPECT_RATIO']").text

        media_uri     = self.encode_uri('media', {'media_id': media_id})
        
        t_media_aspect_ratio_uri = self.transform('aspect_ratio', media_aspect_ratio)
        t_media_video_format     = self.transform('video_format', media_video_format)
        
        self.add_to_graph((media_uri, RDF.type, EBUCore.MediaResource))
        self.add_to_graph((media_uri, EBUCore.aspectRatio, t_media_aspect_ratio_uri))
        self.add_to_graph((media_uri, EBUCore.hasIdentifier, Literal(media_id)))
        self.add_to_graph((media_uri, EBUCore.hasVideoEncodingFormat, t_media_video_format))
        self.add_to_graph((media_uri, EBUCore.frameRate, Literal(media_framerate, datatype=XSD.float)))
        self.add_to_graph((program_uri, EBUCore.isInstanciatedBy, media_uri))

        return media_uri
    
    def add_subtitles_metadata(self, root, program_uri):
        subtitles    = root.findall("./MVAttribute[@type='SUBTITLES']")
        for i, subtitle in enumerate(subtitles):
            subtitles_filename      = subtitle.find("./Meta[@name='ST_FILENAME']").text
            subtitles_language      = subtitle.find("./Meta[@name='ST_LANGUAGE_CODES']").text
            subtitles_file_format   = subtitle.find("./Meta[@name='ST_FILE_FORMAT']").text
            subtitles_date_ingested = subtitle.find("./Meta[@name='ST_INGEST_DATE']").text
            subtitles_date_publised = subtitle.find("./Meta[@name='ST_PUB_DATE']").text

            t_subtitles_language      = self.transform('subtitles_language', subtitles_language)
            t_subtitles_file_format   = self.transform('sub_format', subtitles_file_format)
            t_subtitles_date_ingested = self.transform('date', subtitles_date_ingested)
            t_subtitles_date_publised = self.transform('date', subtitles_date_publised)

            subtitles_uri = self.encode_uri('subtitling', {'n':i, 'program_uri': program_uri})

            self.add_to_graph((subtitles_uri, RDF.type, EBUCore.Subtitling))
            self.add_to_graph((program_uri, EBUCore.hasSubtitling, subtitles_uri))
            self.add_to_graph((subtitles_uri, EBUCore.hasLanguage, t_subtitles_language))
            self.add_to_graph((subtitles_uri, EBUCore.hasFileFormat, t_subtitles_file_format))
            self.add_to_graph((subtitles_uri, EBUCore.dateIngested, t_subtitles_date_ingested))
            self.add_to_graph((subtitles_uri, EBUCore.datePublished, t_subtitles_date_publised))
    
    def add_audio_metadata(self, root, program_uri):
        audios = root.findall("./MVAttribute[@type='AUDIO']")
        for i, audio in enumerate(audios):
            audio_codecs       = audio.find("./Meta[@name='PMA_CODEC']").text
            audio_language     = audio.find("./Meta[@name='PMA_LANGUAGE']").text
            audio_sample_rates = audio.find("./Meta[@name='PMA_SAMPLE_RATE']").text

            t_audio_language = self.transform('audio_language', audio_language)
            
            audio_uri = self.encode_uri('audio', {'n':i, 'program_uri': program_uri})

            self.add_to_graph((audio_uri, RDF.type, EBUCore.AudioTrack))
            self.add_to_graph((program_uri, EBUCore.hasAudioTrack, audio_uri))
            self.add_to_graph((audio_uri, EBUCore.hasLanguage, t_audio_language))
            self.add_to_graph((audio_uri, EBUCore.sampleRate, Literal(audio_sample_rates, datatype=XSD.nonNegativeInteger)))

    def add_pubevents_metadata(self, root, program_uri):
        pubevents    = root.findall("./MVAttribute[@type='PUBLICATIONS']")
        for i, pubevent in enumerate(pubevents):
            pubevent_id           = pubevent.find("./Meta[@name='PUB_ID']").text
            pubevent_datetime     = pubevent.find("./Meta[@name='PUB_DATETIME']").text
            pubevent_channel      = pubevent.find("./Meta[@name='PUB_CHANNEL']").text
            pubevent_datetime_end = pubevent.find("./Meta[@name='PUB_DATETIME_END']").text

            t_pubevent_datetime     = self.transform('datetime', pubevent_datetime)
            t_pubevent_datetime_end = self.transform('datetime', pubevent_datetime_end)
            
            channel_uri  = self.encode_uri('channel', {'name': pubevent_channel})
            self.add_to_graph((channel_uri, RDF.type, EBUCore.PublicationChannel))
            self.add_to_graph((channel_uri, EBUCore.name, Literal(pubevent_channel)))
            
            pubevent_uri = self.encode_uri('publication', {'program_uri': program_uri, 'datetime':pubevent_datetime, 'n': i})
            self.add_to_graph((pubevent_uri, RDF.type, EBUCore.PublicationEvent))
            self.add_to_graph((pubevent_uri, EBUCore.publishes, program_uri))
            self.add_to_graph((pubevent_uri, EBUCore.isReleasedBy, channel_uri))
            self.add_to_graph((pubevent_uri, EBUCore.hasPublicationStartDateTime, t_pubevent_datetime))
            self.add_to_graph((pubevent_uri, EBUCore.hasPublicationEndDateTime, t_pubevent_datetime_end))
    
    def add_segments_metadata(self, root, program_uri):
        # first we get segments descriptions from the corresponding Meta elements and map them to their GUID
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

                    segment_uri = self.encode_uri('segment', {'guid':segment_content_id, 'program_uri': program_uri})

                    t_segment_start = self.transform('time', segment_begin)
                    t_segment_end   = self.transform('time', segment_end)

                    self.add_to_graph((segment_uri, RDF.type, EBUCore.Part))
                    self.add_to_graph((program_uri, EBUCore.hasPart, segment_uri))
                    self.add_to_graph((segment_uri, EBUCore.isPartOf, program_uri))
                    self.add_to_graph((segment_uri, EBUCore.start, t_segment_start))
                    self.add_to_graph((segment_uri, EBUCore.end, t_segment_end))
                    self.add_to_graph((segment_uri, EBUCore.description, Literal(segment_description, lang='fi')))
            else:
                # we have a segment without textual description nor a GUID, we can define a mediaFragment for it?
                pass

    def add_contributors_metadata(self, root, program_uri):
        contributors = root.findall("./MVAttribute[@type='CONTRIBUTORS']")
        for i, contributor in enumerate(contributors):
            contributor_name = contributor.find('./Meta[@name="CONT_PERSON_NAME"]').text
            contributor_role = contributor.find('./Meta[@name="CONT_PERSON_ROLE"]').text
            if contributor_role : contributor_role = contributor_role.strip()

            t_contributor_role = self.transform('contributor_role', contributor_role)

            contributor_uri = self.encode_uri('agent', {'name':contributor_name})

            self.add_to_graph((contributor_uri, RDF.type, EBUCore.Agent))
            self.add_to_graph((program_uri, EBUCore.hasParticipatingAgent, contributor_uri))
            self.add_to_graph((contributor_uri, EBUCore.agentName, Literal(contributor_name)))
            self.add_to_graph((contributor_uri, EBUCore.hasRole, t_contributor_role))

            
    def sparql_query(self, query):
        result = self.graph.query(query)
        return result

    def count_instances_of_type(self, type_uri):
        result = self.graph.query("SELECT DISTINCT ?uri WHERE { ?uri rdf:type "+type_uri+" }")
        return len(result)
