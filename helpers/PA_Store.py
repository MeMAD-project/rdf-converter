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


###########################################
#    INA Professional Archive CSV2RDF     #
###########################################


class PA_Store:
    """ A class for storing and handling INA's Professional Archive CSV datasets """
    """ Constructed from files containing metadata on records on programs and segments """
    """ TV and Radio programs are treated independently """

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
        to_dash = '\\/\',.":;[]()!? #=&$%@{«°»¿=>+*'
        cleaned = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        cleaned = ''.join('-' if c in to_dash else c for c in cleaned)
        cleaned = ''.join(c if i == 0 or (c == '-' and cleaned[i-1]) != '-' else '' for i,c in enumerate(cleaned))
        cleaned = cleaned.lower().strip('-')
        return cleaned
    
    
    def encode_uri(self, resource, data):
        if resource == 'program':
            hashed = sha1(data['id'].encode()).hexdigest() 
            source = data['source'].lower()
            parent = self.clean(data['parent'])
            return URIRef(base + source + '/' + parent + '/' + hashed)
            # return URIRef(base + data['channel'].lower() + '/' + hashed)
        elif resource == 'channel':
            channel_code = self.transform('channel', data['name'])
            return URIRef(base + 'channel/' + channel_code)
        elif resource == 'media':
            hashed = sha1(data['id'].encode()).hexdigest() 
            return URIRef(base + 'media/' + hashed)
        elif resource == 'agent':
            agent_name_cleaned = self.clean(data['name'])
            return URIRef(base + 'agent/' + agent_name_cleaned)
        elif resource in ['timeslot', 'collection']:
            name_cleaned = self.clean(data['name'])
            return URIRef(base + data['source'].lower() + '/' + name_cleaned)
        elif resource == 'publication':
            datetime = ''.join(c for c in data['datetime'] if c in '0123456789')
            n = data['n']
            return URIRef(str(data['program_uri']) + '/publication/' + n)
        elif resource == 'segment':
            hashed = sha1(data['id'].encode()).hexdigest() 
            return URIRef(base + data['source'].lower() + '/' + hashed)
        elif resource == 'record':
            return URIRef(str(data['program_uri']) + '/record')
        else:
            raise Exception('No URI encoding for resource ' + resource)
    
    
    def transform(self, field, value):
        if field == 'duration':
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
            return Literal(date + 'T' + time, datatype=XSD.dateTime)
        elif field == 'date':
            D, M, Y = value.split('/')
            date = Y + '-' + M + '-' + D
            return Literal(date, datatype=XSD.date)
        elif field == 'record_language':
            languages = {'Français': 'French'}
            return Literal(languages[value])
        else:
            raise Exception('No transformation defined for field ' + field + '( value ) :' + str(value))
    
    
    def process_entry(self, entry, autosave=False):
        try:
            # 'Identifiant de la notice' is a common field in PA's "radio" and "tv" metadata, but absent in LD's metadata
            assert('Identifiant de la notice' in entry)  
        except Exception:
            print('The provided file doesn\'t have the appropriate Professional Archive format')

        if "Date d'enregistrement" in entry:
            self.process_program(entry, 'radio_program', autosave)
        else:
            self.process_program(entry, 'tv_program', autosave)
    
    
    def process_program(self, entry, entry_type, autosave=False):
        """ An entry can be either a program record or a segment record """
        channel_uri, channel_code, channel_type = self.add_channel_metadata(entry, entry_type)
        timeslot_uri   = self.add_timeslot_metadata(entry, channel_code)
        collection_uri = self.add_collection_metadata(entry, channel_code, timeslot_uri)
        record_id      = entry['Identifiant de la notice']
        
        program_uri, program_id = self.add_program_metadata(entry, entry_type, channel_type, channel_code, 
                                                            timeslot_uri, collection_uri)

        if record_id.count('_') == 1:
            self.add_pubevent_metadata(entry, program_uri, channel_uri)
            self.add_media_metadata(entry, program_id, program_uri)
        
        self.add_record_metadata(entry, program_uri)
        self.add_contributors_metadata(entry, program_uri)
        self.add_keywords_metadata(entry, program_uri)
        self.add_genres_metadata(entry, program_uri)
        self.add_themes_metadata(entry, program_uri)
        self.add_producers_metadata(entry, program_uri)
               
        if autosave:
            self.save()
    
    
    def add_program_metadata(self, entry, entry_type, channel_type, channel_code, timeslot_uri, collection_uri):
        program_id     = entry['Identifiant de la notice'] # == notice_id
        
        title          = entry['Titre propre'].strip()
        notes          = entry['Notes'].strip()
        summary        = entry['Résumé'].strip().replace('\r', '')
        sequences      = entry['Séquences'].strip().replace('\r', '')
        library        = entry['Thèque'].strip()
        title_notes    = entry['Notes du titre '].strip().replace('\r', '')
        legal_notes    = entry['Notes juridiques'].strip().replace('\r', '')
        corpus         = entry['Corpus  (Aff.)'].strip().replace('\r', '')
        documentalist  = entry['Documentaliste'].strip()
        prod_nature    = entry['Nature de production '].strip()

        correspondent  = entry['Correspondant de chaine']
        last_speaker   = entry['Dernier intervenant']
        external_ref   = entry['Référence extérieure']
        
        ISAN_id = None if 'Numéro ISAN' not in entry else entry['Numéro ISAN']
        lead    = None if 'Chapeau' not in entry else entry['Chapeau'].strip().replace('\r', '')
        producer_summary = None if 'Résumé producteur' not in entry else entry['Résumé producteur'].strip().replace('\r', '')
        
        if collection_uri:
            parent = entry['Titre collection']
        elif timeslot_uri:
            parent = entry['Titre tranche horaire']
        else:
            parent = 'orphan'
        program_uri    = self.encode_uri('program', {'id': program_id, 'source': channel_code, 'parent': parent})
        
        if program_id.count('_') == 2: # this entry is a segment
            source_program_uri = self.encode_uri('program', {'id': program_id[:-4], 'source': channel_code, 'parent': parent})
            self.add_to_graph((source_program_uri, EBUCore.hasPart, program_uri))
            self.add_to_graph((program_uri, EBUCore.isPartOf, source_program_uri))
            self.add_to_graph((program_uri, RDF.type, EBUCore.Part))
        else:
            program_type = EBUCore.TVProgramme if entry_type == 'radio_program' else EBUCore.RadioProgramme
            self.add_to_graph((program_uri, RDF.type, program_type))
        
        self.add_to_graph((program_uri, EBUCore.title, Literal(title)))
        self.add_to_graph((program_uri, EBUCore.summary, Literal(summary, lang='fr')))
        self.add_to_graph((program_uri, RDFS.comment, Literal('Notes: ' + notes if notes else None)))
        self.add_to_graph((program_uri, RDFS.comment, Literal('Notes juridiques: ' + legal_notes if legal_notes else None)))
        self.add_to_graph((program_uri, MeMAD.lead, Literal(lead, lang='fr')))
        self.add_to_graph((program_uri, MeMAD.producerSummary, Literal(producer_summary, lang='fr')))
        self.add_to_graph((program_uri, MeMAD.sequence, Literal(sequences, lang='fr')))
        self.add_to_graph((program_uri, MeMAD.titleNotes, Literal(title_notes, lang='fr')))
        self.add_to_graph((program_uri, MeMAD.hasISANIdentifier, Literal(ISAN_id)))
        # self.add_to_graph((program_uri, MeMAD.productionNature, Literal(prod_nature)))
        # self.add_to_graph((program_uri, MeMAD.library, Literal(library)))
        # self.add_to_graph((program_uri, MeMAD.documentalist, Literal(documentalist)))
        # self.add_to_graph((program_uri, MeMAD.externalReferece, Literal(external_ref)))
        # self.add_to_graph((program_uri, MeMAD.corpus, Literal(corpus)))
        
        if collection_uri:
            self.add_to_graph((collection_uri, EBUCore.isParentOf, program_uri))
            self.add_to_graph((program_uri, EBUCore.isMemberOf, collection_uri))
        
        elif timeslot_uri:
            self.add_to_graph((timeslot_uri, EBUCore.isParentOf, program_uri))
            self.add_to_graph((program_uri, EBUCore.isMemberOf, timeslot_uri))
        
        return program_uri, program_id
    

    def add_timeslot_metadata(self, entry, channel_code):
        timeslot_name = entry['Titre tranche horaire']
        
        if timeslot_name == '': 
            return None
        
        timeslot_uri = self.encode_uri('timeslot', {'name': timeslot_name, 'source':channel_code})
        
        self.add_to_graph((timeslot_uri, RDF.type, MeMAD.Timeslot))
        self.add_to_graph((timeslot_uri, EBUCore.title, Literal(timeslot_name)))  
        return timeslot_uri
    
    
    def add_collection_metadata(self, entry, channel_code, timeslot_uri):
        collection_name = entry['Titre collection']
        
        if collection_name == '':
            return None
        
        collection_uri = self.encode_uri('collection', {'name': collection_name, 'source':channel_code})
        
        self.add_to_graph((collection_uri, RDF.type, EBUCore.Collection))
        self.add_to_graph((collection_uri, EBUCore.title, Literal(collection_name)))
        
        if timeslot_uri:
            self.add_to_graph((timeslot_uri, EBUCore.isParentOf, collection_uri))
            self.add_to_graph((collection_uri, EBUCore.isMemberOf, timeslot_uri))
        
        return collection_uri
    
    
    def add_channel_metadata(self, entry, entry_type):
        channel_name = entry['Canal de diffusion']
        
        channel_uri    = self.encode_uri('channel', {'name': channel_name})
        t_channel_code = self.transform('channel', channel_name)
        channel_type   = 'Radio' if entry_type == 'radio_program' else 'TV'
      
        self.add_to_graph((channel_uri, RDF.type, EBUCore.PublicationChannel))
        self.add_to_graph((channel_uri, EBUCore.publicationChannelId, Literal(t_channel_code)))
        self.add_to_graph((channel_uri, EBUCore.publicationChannelName, Literal(channel_name)))
        self.add_to_graph((channel_uri, EBUCore.serviceDescription, Literal(channel_type + ' channel')))

        return channel_uri, t_channel_code, channel_type
    
    
    def add_record_metadata(self, entry, program_uri):
        record_id            = entry['Identifiant de la notice']
        record_creation_date = entry['Date de création']
        record_update_date   = entry['Date de modification']
        record_language      = entry['Langue de la notice']
        record_type          = entry['Type de notice']
        
        record_uri = self.encode_uri('record', {'program_uri': program_uri})
        
        t_record_language = self.transform('record_language', record_language)
        t_creation_date = self.transform('date', record_creation_date)
        t_update_date   = self.transform('date', record_update_date)
        
        self.add_to_graph((record_uri, RDF.type, MeMAD.Record))
        self.add_to_graph((program_uri, MeMAD.hasRecord, record_uri))
        self.add_to_graph((record_uri, EBUCore.dateCreated, t_creation_date))
        self.add_to_graph((record_uri, EBUCore.dateUpdated, t_update_date))
        self.add_to_graph((record_uri, EBUCore.language, t_record_language))
        self.add_to_graph((record_uri, EBUCore.hasType,  Literal(record_type)))
    
    
    def add_media_metadata(self, entry, program_id, program_uri):
        media_uri     = self.encode_uri('media', {'id': program_id})

        material_id    = entry['Identifiant Matériels' if 'Identifiant Matériels' in entry else 'Identifiant Matériels (info.)']
        material_id    = material_id.strip().replace('\r', '')
        material_note  = None if 'Matériels  (Détail)' not in entry else entry['Matériels  (Détail)'].strip().replace('\r', '')

        self.add_to_graph((media_uri, RDF.type, EBUCore.MediaResource))
        self.add_to_graph((program_uri, EBUCore.isInstanciatedBy, media_uri))
        self.add_to_graph((media_uri, RDFS.comment, Literal('Identifiant Matériels: ' + material_id if material_id else None)))
        self.add_to_graph((media_uri, RDFS.comment, Literal('Matériels  (Détail): ' + material_note if material_note else None)))
    
    
    def add_contributors_metadata(self, entry, program_uri):
        # a credit = "PRE Atlan, Monique  ;"
        credits    = entry['Générique (Aff. Lig.) '].strip().split(';')
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
                
                agent_uri = self.encode_uri('agent', {'name': name})

                self.add_to_graph((agent_uri, RDF.type, EBUCore.Agent))
                self.add_to_graph((agent_uri, EBUCore.hasRole, Literal(role, lang='fr')))
                self.add_to_graph((agent_uri, EBUCore.agentName, Literal(name)))
                self.add_to_graph((program_uri, EBUCore.hasParticipatingAgent, agent_uri))
    
    
    def add_keywords_metadata(self, entry, program_uri):
        keywords   = entry['Descripteurs (Aff. Lig.)'].strip().split(';')
        for keyword in keywords:
            keyword = keyword.strip()
            if keyword: 
                keyword = keyword[4:].strip()
                self.add_to_graph((program_uri, EBUCore.hasKeyword, Literal(keyword)))
    
    
    def add_genres_metadata(self, entry, program_uri):
        genres     = entry['Genre'].strip().split(';')
        for genre in genres:
            genre = genre.strip()
            if genre : 
                self.add_to_graph((program_uri, EBUCore.hasGenre, Literal(genre)))
    
    
    def add_producers_metadata(self, entry, program_uri):
        producers  = entry['Producteurs (Aff.)'].strip().replace('\r', '').split('\n')
        for producer in producers:
            producer = producer.strip()
            if producer.strip() : 
                self.add_to_graph((program_uri, EBUCore.hasProducer, Literal(producer)))
    
    
    def add_themes_metadata(self, entry, program_uri):
        themes     = entry['Thématique'].strip().split(';')
        for theme in themes:
            theme = theme.strip()
            if theme : 
                self.add_to_graph((program_uri, EBUCore.hasTheme, Literal(theme)))
    
    
    def add_pubevent_metadata(self, entry, program_uri, channel_uri):
        broadcast_date = entry['Date de diffusion']
        broadcast_time = '00:00:00' if 'Heure de diffusion' not in entry else entry['Heure de diffusion']
        geo_scope      = entry['Extension géographique (info.)']
        duration       = entry['Durée']
        
        t_duration       = self.transform('duration', duration)
        t_broadcast_date = self.transform('datetime', broadcast_date+broadcast_time)

        pubevent_uri = self.encode_uri('publication', {'program_uri': program_uri, 'datetime':broadcast_date+broadcast_time, 'n': '0'})
        
        self.add_to_graph((pubevent_uri, EBUCore.hasPublicationStartDateTime, t_broadcast_date))
        self.add_to_graph((pubevent_uri, RDF.type, EBUCore.PublicationEvent))
        self.add_to_graph((pubevent_uri, EBUCore.publishes, program_uri))
        self.add_to_graph((pubevent_uri, EBUCore.isReleasedBy, channel_uri))
        self.add_to_graph((pubevent_uri, EBUCore.duration, t_duration))
        self.add_to_graph((pubevent_uri, EBUCore.hasPublicationRegion, Literal(geo_scope)))

    
    def sparql_query(self, query):
        result = self.graph.query(query)
        return result
    
    
    def count_instances_of_type(self, type_uri):
        result = self.graph.query("SELECT DISTINCT ?uri WHERE { ?uri rdf:type "+type_uri+" }")
        return len(result)
