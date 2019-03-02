# -*- coding: utf-8 -*-

import time
import unicodedata
import json
import os
import rdflib
from os.path import exists, dirname
from hashlib import sha1
from rdflib import Namespace, URIRef, ConjunctiveGraph, Literal
from rdflib.namespace import FOAF, DC, SKOS, RDF, RDFS, XSD
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


# Declaring all Namespaces
INA     = Namespace('http://www.ina.fr/core#')
Yle     = Namespace('http://www.yle.fi/ontology#')
MeMAD   = Namespace('http://data.memad.eu/ontology#')
EBUCore = Namespace('http://www.ebu.ch/metadata/ontologies/ebucore/ebucore#')

base = 'http://data.memad.eu/'


####################################
#    INA Legal Deposit CSV2RDF     #
####################################


class LD_Store:
    """ A class for storing and handling INA's Legal Deposit CSV datasets """
    """ Constructed from files containing metadata on records on programs and segments """
    """ Programs and Segments are treated independently """

    def __init__(self, store_path = None):
        """ Load a graph from a given path or create a blank one + bind namespaces to prefixes """
        # creating or loading the in-memory graph
        self.graph = ConjunctiveGraph()
        self.graph.bind('skos', SKOS)
        self.graph.bind('memad', MeMAD)
        self.graph.bind('ebucore', EBUCore)
        
        self.path  = store_path
        if self.path and exists(self.path):
            self.graph.load(self.path, format='turtle')
        elif not exists(dirname(self.path)):
            print('Creating directory :' + dirname(self.path))
            os.makedirs(dirname(self.path))
    
    
    def get_graph(self):
        return self.graph
        

    def save(self, save_path = None):
        """ serialize the graph into Turtle (to different path if the argument is given) """
        self.graph.serialize(save_apth if save_path else self.path, format='turtle')
    
    
    def add_to_graph(self, triplet, signal_empty_values=False):
        if triplet[2] and str(triplet[2]) != 'None': # the predicate has a non-null value
            self.graph.add(triplet)
        elif signal_empty_values:
            print(str(triplet[1]) + '.' + str(triplet[1]) + ' was not added to the graph (empty value)')
    
    
    def clean(self, s):
        """ Transforming free text strings into ascii slugs """
        to_dash = '\\/\',.":;[]()!? #=&$%@{«°»¿=>+*'
        cleaned = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        cleaned = ''.join('-' if c in to_dash else c for c in cleaned)
        cleaned = ''.join(c if i == 0 or (c == '-' and cleaned[i-1]) != '-' else '' for i,c in enumerate(cleaned))
        cleaned = cleaned.lower().strip('-')
        return cleaned
    
    
    def encode_uri(self, resource, data):
        """ Generating URIs for resources """
        if resource == 'program':
            hashed = sha1(data['id'].encode()).hexdigest() 
            source = data['source'].lower()
            parent = self.clean(data['parent'])
            return URIRef(base + source + '/' + parent + '/' + hashed)
            # return URIRef(base + data['channel'].lower() + '/' + hashed)
        elif resource == 'channel':
            channel_code = self.transform('channel', data['name']).lower()
            return URIRef(base + 'channel/' + channel_code)
        elif resource == 'media':
            hashed = sha1(data['id'].encode()).hexdigest() 
            return URIRef(base + 'media/' + hashed)
        elif resource == 'agent':
            agent_name_cleaned = self.clean(data['name'])
            return URIRef(base + 'agent/' + agent_name_cleaned)
        elif resource in ['timeslot', 'collection']:
            name_cleaned = self.clean(data['name'])
            source = data['channel'].lower()
            return URIRef(base + source + '/' + name_cleaned)
        elif resource == 'history':
            return URIRef(str(data['program_uri']) + '/publication')
        elif resource == 'publication':
            datetime = data['datetime'].replace(' ', '').replace('-', '').replace(':', '')
            n = data['n']
            return URIRef(str(data['program_uri'] + '/publication/' + n))
        elif resource == 'segment':
            hashed = sha1(data['id'].encode()).hexdigest() 
            source = data['channel'].lower()
            return URIRef(base + source + '/' + hashed)       
        else:
            raise Exception('No URI encoding for resource ' + resource)
    
    
    def transform(self, field, value):
        if field == 'duration':
            duration = int(value)
            result = 'PT' + (str(int(duration / 3600)).zfill(2) + 'H' if duration >= 3600 else '')
            duration -= int(duration / 3600) * 3600
            result += (str(int(duration / 60)).zfill(2) + 'M') if duration >= 60 else ''
            duration -= int(duration / 60) * 60
            result += (str(int(duration)).zfill(2) + 'S')
            return Literal(result, datatype=XSD.duration)
        elif field == 'channel':
            channel_codes = json.load(open('mappings/ina_channel2code.json'))
            return channel_codes[value]
        elif field == 'datetime':
            # "2014-05-01T05:33:17+01:00" in Emissions  or  "2014-05-01 05:33:17" in Segments
            result = value.replace(' ', 'T')[:19]
            return Literal(result, datatype=XSD.dateTime)
        else:
            raise Exception('No transformation defined for field ' + field)
    

    def process_entry(self, entry, autosave=False):
        try:
            # 'Identidiant' is a common field in LD's "emmision" and "sujet" metadata, but absent in PA's metadata
            assert('Identifiant' in entry)  
        except Exception:
            print('The provided file doesn\'t have the appropriate Legal Deposit format')

        if 'referenceDate' in entry:
            self.process_program_entry(entry, autosave)
        else:
            self.process_segment_entry(entry, autosave)


    def process_program_entry(self, entry, autosave=False):
        channel_uri, channel_code, channel_type = self.add_channel_metadata(entry)
        timeslot_uri   = self.add_timeslot_metadata(entry, channel_code)
        collection_uri = self.add_collection_metadata(entry, channel_code, timeslot_uri)
        
        program_uri, program_id = self.add_program_metadata(entry, channel_type, channel_code, timeslot_uri, collection_uri)
        
        self.add_pubevent_metadata(entry, program_uri, channel_uri)
        self.add_media_metadata(entry, program_id, program_uri)
        self.add_contributors_metadata(entry, program_uri)
        self.add_keywords_metadata(entry, program_uri)
        self.add_genres_metadata(entry, program_uri)
        self.add_themes_metadata(entry, program_uri)
        self.add_producers_metadata(entry, program_uri)
               
        if autosave:
            self.save()
    
     
    def process_segment_entry(self, entry, autosave=False):
        assert(entry['Identifiant'].count('_') == 2)
        program_id   = entry['Identifiant'][:-4]
        timeslot_name   = entry['TitreTrancheHoraire']
        collection_name = entry['TitreCollection']

        if collection_name:
            parent = collection_name
        elif timeslot_name:
            parent = timeslot_name
        else:
            parent = 'orphan'

        channel_uri, channel_code, channel_type = self.add_channel_metadata(entry)

        program_uri = self.encode_uri('program', {'id': program_id, 'source': channel_code, 'parent':  parent})
        segment_uri = self.add_segment_metadata(entry, program_uri, channel_code)

        self.add_contributors_metadata(entry, segment_uri)
        self.add_keywords_metadata(entry, segment_uri)
               
        if autosave:
            self.save()
    
    
    def add_program_metadata(self, entry, channel_type, channel_code, timeslot_uri, collection_uri):
        program_id       = entry['Identifiant'] # == notice_id
        channel_name     = entry['Chaine']
        duration         = entry['DureeSecondes']
        title            = entry['TitreEmission']
        summary          = entry['Resume'].strip().replace('\r', '')
        lead             = entry['Chapeau'].strip().replace('\r', '')
        note_dispositif  = entry['Dispositif'].strip().replace('\r', '')
        prod_nature      = entry['NatureProduction']
        producer_summary = entry['ResumeProducteur'].strip().replace('\r', '')

        
        if collection_uri:
            parent = entry['TitreCollection']
        elif timeslot_uri:
            parent = entry['TitreTrancheHoraire']
        else:
            parent = 'orphan'
        program_uri = self.encode_uri('program', {'id': program_id, 'parent':parent, 'source': channel_code})
        
        t_duration     = self.transform('duration', duration)
        program_type   = EBUCore.RadioProgramme if channel_type == 'Radio' else EBUCore.TVProgramme
        
        self.add_to_graph((program_uri, RDF.type, program_type))
        self.add_to_graph((program_uri, EBUCore.title, Literal(title)))
        self.add_to_graph((program_uri, EBUCore.duration, t_duration))
        self.add_to_graph((program_uri, EBUCore.summary, Literal(summary, lang='fr')))
        self.add_to_graph((program_uri, MeMAD.lead, Literal(lead, lang='fr')))
        self.add_to_graph((program_uri, RDFS.comment, Literal('Dispositif: ' + note_dispositif if note_dispositif else '')))
        # self.add_to_graph((program_uri, MeMAD.productionNature, Literal(prod_nature)))
        
        
        if collection_uri:
            self.add_to_graph((collection_uri, EBUCore.isParentOf, program_uri))
            # self.add_to_graph((program_uri, EBUCore.isMemberOf, collection_uri))
        
        elif timeslot_uri:
            self.add_to_graph((timeslot_uri, EBUCore.isParentOf, program_uri))
            # self.add_to_graph((program_uri, EBUCore.isMemberOf, timeslot_uri))
        
        return program_uri, program_id
    
    
    def add_media_metadata(self, entry, program_id, program_uri):
        Imedia_id        = entry['IdentifiantImedia']
        Mediametrie_id   = entry['IdentifiantMediametrie']

        media_uri     = self.encode_uri('media', {'id': program_id})

        self.add_to_graph((media_uri, RDF.type, EBUCore.MediaResource))
        self.add_to_graph((program_uri, EBUCore.isInstantiatedBy, media_uri))
    
        self.add_to_graph((media_uri, MeMAD.hasImediaIdentifier, Literal(Imedia_id)))
        self.add_to_graph((media_uri, MeMAD.hasMediametrieIdentifier, Literal(Mediametrie_id)))
       
    
    def add_contributors_metadata(self, entry, program_uri):
        credits    = entry['Generiques' if 'Generiques' in entry else 'Generique'].strip().split('|')
        for credit in credits:
            if credit == '': continue
            if '#' in credit : uid, credit = credit.split('#')
            if '(' in credit : name, role = credit.split('(')

            role = role.strip()[:-1]
            name = name.strip()
            agent_uri = self.encode_uri('agent', {'name': name})

            self.add_to_graph((agent_uri, RDF.type, EBUCore.Agent))
            self.add_to_graph((agent_uri, EBUCore.agentName, Literal(name)))
            self.add_to_graph((agent_uri, EBUCore.hasRole, Literal(role, lang='fr')))
            self.add_to_graph((program_uri, EBUCore.hasParticipatingAgent, agent_uri))
    
    
    def add_timeslot_metadata(self, entry, channel_code):
        timeslot_name = entry['TitreTrancheHoraire']
        
        if timeslot_name == '': 
            return None
        
        timeslot_uri = self.encode_uri('timeslot', {'name': timeslot_name, 'channel':channel_code})
        
        self.add_to_graph((timeslot_uri, RDF.type, MeMAD.Timeslot))
        self.add_to_graph((timeslot_uri, EBUCore.title, Literal(timeslot_name)))  
        return timeslot_uri
    
    
    def add_collection_metadata(self, entry, channel_code, timeslot_uri):
        collection_name = entry['TitreCollection']
        
        if collection_name == '':
            return None
        
        collection_uri = self.encode_uri('collection', {'name': collection_name, 'channel':channel_code})
        
        self.add_to_graph((collection_uri, RDF.type, EBUCore.Collection))
        self.add_to_graph((collection_uri, EBUCore.title, Literal(collection_name)))
        
        if timeslot_uri:
            self.add_to_graph((timeslot_uri, EBUCore.isParentOf, collection_uri))
            # self.add_to_graph((collection_uri, EBUCore.isMemberOf, timeslot_uri))
        
        return collection_uri
    
    
    def add_channel_metadata(self, entry):
        channel_name = entry['Chaine']
        radio_channels = set(['BEU' , 'BFM', 'CHE', 'D8_', 'EU1', 'MUV', 'GA1', 'EU2', 'FBL', 'FCR', 
                              'FIF', 'FIT', 'FMU', 'FUN', 'MUV', 'NOS', 'NRJ' , 'RBL', 'RCL' , 'RFI',
                              'RFM', 'RIR', 'RMC', 'RT2', 'RTL', 'RT9', 'SKY', 'SUD', 'VIR'])
        
        channel_uri    = self.encode_uri('channel', {'name': channel_name})
        t_channel_code = self.transform('channel', channel_name)
        channel_type   = 'Radio' if t_channel_code in radio_channels else 'TV'
        
        self.add_to_graph((channel_uri, RDF.type, EBUCore.PublicationChannel))
        self.add_to_graph((channel_uri, EBUCore.publicationChannelId, Literal(t_channel_code)))
        self.add_to_graph((channel_uri, EBUCore.publicationChannelName, Literal(channel_name)))
        self.add_to_graph((channel_uri, EBUCore.serviceDescription, Literal(channel_type + ' channel')))

        return channel_uri, t_channel_code, channel_type
    
    
    def add_keywords_metadata(self, entry, program_uri):
        keywords   = entry['Descripteurs'].strip().split('|')
        for keyword in keywords:
            if keyword.strip() : self.add_to_graph((program_uri, EBUCore.hasKeyword, Literal(keyword)))
    
    
    def add_genres_metadata(self, entry, program_uri):
        genres     = entry['Genres'].strip().split('|')
        for genre in genres:
            if genre.strip() : self.add_to_graph((program_uri, EBUCore.hasGenre, Literal(genre)))
    
    
    def add_producers_metadata(self, entry, program_uri):
        producers  = entry['Producteurs'].strip().split('|')
        for producer in producers:
            if producer.strip() : self.add_to_graph((program_uri, EBUCore.hasProducer, Literal(producer)))
    
    
    def add_themes_metadata(self, entry, program_uri):
        themes     = entry['Thematique'].strip().split('|')
        for theme in themes:
            if theme.strip() : self.add_to_graph((program_uri, EBUCore.hasTheme, Literal(theme)))
    
    
    def add_pubevent_metadata(self, entry, program_uri, channel_uri):
        pubevent_channel      = entry['Chaine']
        pubevent_datetime     = entry['startDate']
        pubevent_datetime_end = entry['endDate']
        reference_date        = entry['referenceDate']

        t_pubevent_datetime     = self.transform('datetime', pubevent_datetime)
        t_pubevent_datetime_end = self.transform('datetime', pubevent_datetime_end)

        history_uri = self.encode_uri('history', {'program_uri': program_uri})
        pubevent_uri = self.encode_uri('publication', {'program_uri': program_uri, 'datetime':pubevent_datetime, 'n': '0'})

        self.add_to_graph((history_uri, RDF.type, EBUCore.PublicationHistory))
        self.add_to_graph((program_uri, EBUCore.hasPublicationHistory, history_uri))
        self.add_to_graph((history_uri, EBUCore.hasPublicationEvent, pubevent_uri))

        self.add_to_graph((pubevent_uri, RDF.type, EBUCore.PublicationEvent))
        self.add_to_graph((pubevent_uri, EBUCore.publishes, program_uri))
        self.add_to_graph((pubevent_uri, EBUCore.isReleasedBy, channel_uri))
        self.add_to_graph((pubevent_uri, EBUCore.hasPublicationStartDateTime, t_pubevent_datetime))
        self.add_to_graph((pubevent_uri, EBUCore.hasPublicationEndDateTime, t_pubevent_datetime_end))
        # self.add_to_graph((program_uri, MeMAD.referenceDate, Literal(reference_date)))
    
    
    def add_segment_metadata(self, entry, program_uri, channel_code):
        segment_id = entry['Identifiant']
        program_id = entry['Identifiant'][:-4]
        start_date = entry['startDate']
        duration   = entry['DureeSecondes']
        lead       = entry['Chapeau']
        segment_title = entry['TitreSujet']
        program_title = entry['TitreEmission']

        timeslot_uri   = self.add_timeslot_metadata(entry, channel_code)
        collection_uri = self.add_collection_metadata(entry, channel_code, timeslot_uri)
        
        segment_uri = self.encode_uri('segment', {'id': segment_id, 'channel': channel_code})
        
        t_start_date = self.transform('datetime', start_date) if duration else None
        t_duration   = self.transform('duration', duration) if duration else None
        
        # Adding entities to the graph
        self.add_to_graph((program_uri, EBUCore.hasPart, segment_uri))
        self.add_to_graph((program_uri, EBUCore.title, Literal(program_title)))
        self.add_to_graph((segment_uri, RDF.type, EBUCore.Part))
        self.add_to_graph((segment_uri, EBUCore.title, Literal(segment_title)))
        self.add_to_graph((segment_uri, EBUCore.isPartOf, program_uri))
        self.add_to_graph((segment_uri, EBUCore.isPartOf, program_uri))
        self.add_to_graph((segment_uri, EBUCore.start, t_start_date))
        self.add_to_graph((segment_uri, EBUCore.duration, t_duration))
        self.add_to_graph((segment_uri, MeMAD.lead, Literal(lead, lang='fr')))
        
        return segment_uri
        
    
    def sparql_query(self, query):
        result = self.graph.query(query)
        return result
    
    
    def count_instances_of_type(self, type_uri):
        result = self.graph.query("SELECT DISTINCT ?uri WHERE { ?uri rdf:type "+type_uri+" }")
        return len(result)
