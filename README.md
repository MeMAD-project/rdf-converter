# rdf-converter
MeMAD metadata converter that transforms legacy metadata from INA and Yle into RDF using the MeMAD and EBU Core ontologies.
MeMAD Ontology: <http://data.memad.eu/ontology>

### Prerequisites

All the scripts are written in Python3. To run these scripts, the following libraries should be installed beforehand: [pandas](https://pandas.pydata.org/), [tqdm](https://github.com/tqdm/tqdm) and [rdflib](https://rdflib.readthedocs.io/en/stable/). These dependecies can be installed using `pip`.


### Running the script

```
usage: [ld|pa|yle]_converter.py [-h] [-p PATH] [-o OUTPUT] [-f FLOW_MAPPING] [-k]

MeMAD Converter

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Specify the path for the dataset (or datasets) to process
  -o OUTPUT, --output OUTPUT
                        Specify the path to which the TTL output would be
                        written.
  -f FLOW_MAPPING, --flow_mapping FLOW_MAPPING
                        Specify the path to a file containing the mapping
                        between filenames and their Flow identifier.
  -k, --keep_mappings   add this flag to generate CSV files for mapping
                        Programs to their URIs
```
