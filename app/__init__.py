from flask import Flask

import gensim
import re

from gensim.models import LdaModel
from gensim.test.utils import datapath
from gensim import corpora, models
from gensim.corpora import Dictionary

from re import sub
import os
import string
import codecs

import nltk
#from nltk.tokenize import RegexpTokenizer
#from nltk import stem
#from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
from app import views
from app import model