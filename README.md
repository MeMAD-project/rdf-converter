# rdf-converter
MeMAD metadata converter that transforms legacy metadata from INA and Yle into RDF using the MeMAD and EBU Core ontologies

### Prerequisites

All the scripts are written in Python3. To run these scripts, the following libraries should be installed beforehand: [pandas](https://pandas.pydata.org/) and [rdflib](https://rdflib.readthedocs.io/en/stable/). These dependecies can be installed using `pip`.


### Running the script

```
usage: memad-converter.py [-h] -p PATH -d {ina-ld,ina-pa,yle}
MeMAD Converter

optional arguments:
 -h, --help            show this help message and exit
 -p PATH, --path PATH  specify the path for the file or folder to process
 -d {ina-ld,ina-pa,yle}, --dataset {ina-ld,ina-pa,yle}
                       specify which converter to use.
```
