from app import app
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

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

import pandas as pd

import nltk

#from nltk.tokenize import RegexpTokenizer
from nltk import stem
from nltk.stem import WordNetLemmatizer

def remove_shorts (words):
    words = re.sub(r'\b\w{1,3}\b', '', words)
    return words

def remove_special_characters(words, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    words = re.sub(pattern, '', words)
    return words

def normalize(words):
    words = remove_shorts(words)
    words = remove_special_characters(words, remove_digits=True)
    return words

def create_stopwords(path):
    stop_words = []
    for w in open(path, "r",encoding="utf-8"):
        w = w.replace('\n','')
        if len(w) > 0:
             stop_words.append(w)
    return stop_words

absFilePath = os.path.abspath(__file__)                # Absolute Path of the module

fileDir = os.path.dirname(os.path.abspath(__file__))   # Directory of the Module

parentDir = os.path.dirname(fileDir)                   # Directory of the Module directory

modelPath = os.path.join(fileDir, 'model')             # Get the directory for model

filePath = os.path.join(fileDir, 'upload')              # Directory of the test file

swPath = os.path.join(fileDir, 'files')

temp_file = datapath((os.path.join(modelPath, 'LDAmodel_K17')))
lda_model = LdaModel.load(datapath(temp_file))

# create stop_words
temp_file = datapath((os.path.join(swPath, 'JPSSM_project.txt')))
stop_words = create_stopwords(temp_file)

# load dictionary
original = Dictionary.load((os.path.join(modelPath, 'LDAmodel_K17.id2word')))

# load full table
full_table = pd.read_csv(os.path.join(swPath, 'LDA_table_complete_numbers.tsv'), sep='\t')


ENV = ''

if ENV == 'dev':
    app.debug = True
    app.config['SQLALCHEMY_DATABASE_URI'] = ''
else:
    app.debug = False
    app.config['SQLALCHEMY_DATABASE_URI'] = ''

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

@app.route("/")
def home():
    return "Welcome page <h1>hello</h1>"

@app.route("/upload-file", methods=["GET", "POST"])
def upload_file():
    return render_template("public/upload_file.html")

@app.route("/text-submit/", methods=["GET", "POST"])
def text_submit():
    if request.method=='GET':
        return render_template('/text_submit.html')
    elif request.method=='POST':
        user_txt = request.form.get('user_txt')

        test_file = user_txt

        # prepare the Text
        # clean up text: rem white space, new line marks, blank lines
        body_text = test_file.strip().replace('  ', ' ')
        body_text = body_text.replace('\n', ' ').replace('\r', '')

            # delete digits
        body_text = sub(pattern=r"\d", repl=r" ", string=body_text)

            # remove punctuation - updated
        translator = str.maketrans(' ',' ', string.punctuation)
        body_text = body_text.translate(translator)
        body_text = os.linesep.join([s for s in body_text.splitlines() if s])

        lemmatizer = stem.WordNetLemmatizer()
        test_text=[]
        raw = body_text.lower()
        raw = normalize(raw)

        tokens = nltk.word_tokenize(raw)
        stopped_tokens = [i for i in tokens if not i in stop_words]
        lemmatized_tokens = [lemmatizer.lemmatize(i) for i in stopped_tokens]
        test_text.extend(lemmatized_tokens)

        raw = raw.lower()
        raw = raw.split()
        vec_bow = original.doc2bow(raw)
        vec_lda = lda_model[vec_bow]
        result=dict(vec_lda)

        topic_names2 = {"0":"Econometric models of performance","1":"Commitment-Trust in Sales","2":"Diversity and inclusion in sales","3":"Sales technology & systems 1.0","4":"Compensation / Bonus","5":"Customer orientation","6":"Scales in sales research","7":"Sales technology & systems 2.0","8":"Research methods","9":"Salesperson interpersonal communication","10":"Sales teams","11":"Salespeople turnover","12":"International sales","13":"Salesperson's evaluation and training","14":"Salesperson's success and failure","15":"Buyer-seller relationship", "16":"Leadership and adaptive selling"}
        topic_names = {0:"Econometric models of performance",1:"Commitment-Trust in Sales",2:"Diversity and inclusion in sales",3:"Sales technology & systems 1.0",4:"Compensation / Bonus",5:"Customer orientation",6:"Scales in sales research",7:"Sales technology & systems 2.0",8:"Research methods",9:"Salesperson interpersonal communication",10:"Sales teams",11:"Salespeople turnover",12:"International sales",13:"Salesperson's evaluation and training",14:"Salesperson's success and failure",15:"Buyer-seller relationship", 16:"Leadership and adaptive selling"}
        
        from collections import OrderedDict
        result = OrderedDict(sorted(result.items(), key=lambda x: x[1], reverse = True))
        
        # get top 3 keys
        top_keys=[]
        for z in range(3):
            r=list(result.keys())[z]
            r=str(r)
            top_keys.append(r)

        # Tables with top results

        # Table 1
        key_1 = top_keys[0]

        # table preparation
        columns_to_show = ['Title','Authors','Year']
        columns_to_show.extend(key_1)

        new_table = full_table[columns_to_show]
        to_display = new_table.nlargest(10, key_1)
        
        # name mapping
        
        to_display = to_display.rename(topic_names2, axis='columns')
        to_display_dict = to_display.to_dict()
        
        records = to_display.to_records(index=False)
        result_records = list(records)

        # table 2
        
        key_2 = top_keys[1]
        key_2=str(key_2)
        # table preparation

        columns_to_show2 = ['Title','Authors','Year']
        
        columns_to_show2.append(key_2)

        new_table2 = full_table[columns_to_show2]
        
        to_display2 = new_table2.nlargest(10, top_keys[1])
        
        # name mapping

        to_display2 = to_display2.rename(topic_names2, axis='columns')
        to_display_dict2 = to_display2.to_dict()
        
        records2 = to_display2.to_records(index=False)
        result_records2 = list(records2)


        # table 3
        
        key_3 = top_keys[2]
        key_3=str(key_3)
        # table preparation
        
        columns_to_show3 = ['Title','Authors','Year']
        
        columns_to_show3.append(key_3)

        new_table3 = full_table[columns_to_show3]
        
        to_display3 = new_table3.nlargest(10, top_keys[2])
        
        # name mapping
        to_display3 = to_display3.rename(topic_names2, axis='columns')
        to_display_dict3 = to_display3.to_dict()
        
        records3 = to_display3.to_records(index=False)
        result_records3 = list(records3)
        

        # the main table
        dane=[]
        for x in result.values():
            dane.append("{:.2f}".format(x))


        def map_fields(init_dict, map_dict, res_dict=None):
            res_dict = res_dict or {}
            for k, v in init_dict.items():
                if isinstance(v, dict):
                    v = map_fields(v, map_dict)
                if k in map_dict.keys():
                    k = str(map_dict[k])
                res_dict[k] = v

            return res_dict

       
        result_dict = map_fields(result, topic_names)


        result = {}
        result = result_dict

    return render_template('text_submit.html', result=result, dane=dane, to_display_dict=to_display_dict,to_display_dict2=to_display_dict2,to_display_dict3=to_display_dict3 ,result_records=result_records, result_records2=result_records2, result_records3=result_records3)

@app.route("/result/")
def result():
    topic_names2 = {"0":"Econometric models of performance","1":"Commitment-Trust in Sales","2":"Diversity and inclusion in sales","3":"Sales technology & systems 1.0","4":"Compensation / Bonus","5":"Customer orientation","6":"Scales in sales research","7":"Sales technology & systems 2.0","8":"Research methods","9":"Salesperson interpersonal communication","10":"Sales teams","11":"Salespeople turnover","12":"International sales","13":"Salesperson's evaluation and training","14":"Salesperson's success and failure","15":"Buyer-seller relationship", "16":"Leadership and adaptive selling"}
    to_display = full_table.rename(topic_names2, axis='columns')
    
    headings = list(to_display.columns)
    
    records = to_display.to_records(index=False)
    result_records = list(records)
    
    
    return render_template('results.html', results_records=result_records, headings=headings)

if __name__ == '__main__':
    app.run()
