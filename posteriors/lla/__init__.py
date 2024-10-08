import os

PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__)) # Gets directory name where this file resides
ROOT = '/'.join(PACKAGE_DIR.split('/')[:-1]) 
DATA_DIR = ROOT + '/data'
