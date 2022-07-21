# Machine Learning Approach for Conducting Sales Literature Reviews: An Application to the Forty Years of JPSSM


This repository hosts the code and files used in preparing the manuscript. The code is provided in this very document and divided into separate segments which follow the Figure 2 (same figuer as in this repository). The code can be pasted to Google Collaboratory (https://colab.research.google.com) and run there step by step. If you want to run it on your own machine, we would recommend using [Anaconda](https://docs.anaconda.com/anaconda/install/) and Jupter Notebooks.

How to cite the work:
```bash
Hideaki Kitanaka, Piotr Kwiatek, Nikolaos G Panagopoulos. (2021). Introducing a New, Machine Learning Process, and Online Tools for Conducting Sales Literature Reviews: An Application to the Forty Years of JPSSM, Journal of Personal Selling and Sales Management 41(4), 351-368.
Doi: 10.1080/08853134.2021.1935976
```

The effects of the project (interactive charts & application) are available on our [website](https://salesai.online). The application showcases the use of classification model and follows Figure 2 from the paper.
If you would like to run the application locally on your computer, fork and clone the repository. Have a look at friendly [GitHub tutorials](https://guides.github.com/activities/hello-world/) if you are a novice to this.

Once you have the repository on your local machine, you need to set up Python environment. Have a look at this short [tutorial](https://docs.python-guide.org/dev/virtualenvs/) - it offers step by step congfiguration of the environment (apply the steps from 'Lower level: virtualenv' and you can stop at 'Other Notes'). 

After setting up the environment, you can run the application from the terminal:
```bash
flask run
```
Next, open your browser and type:
```bash
localhost:5000/text-submit
```

Voila!


## The beginning
The project starts with collecting the PDF version of articles we want to analyze. Create a folder to store the documents (_PATH_TO_PDFs_). In the next step we will convert them to XML versions.
## Requirements

There are two types of requirements used: (1) Software, and (2) Python packages (and libraries). The software is listed below and Python packages are documented in: 

```bash
requirements.txt
```
Software used:
1. For PDF extraction you can use [CERMINE](http://cermine.ceon.pl/index.html) solution. Just download the package (java jar) and execute it at the prompt: 
```bash
java -cp cermine-impl-1.13-jar-with-dependencies.jar pl.edu.icm.cermine.ContentExtractor -path /PUT_THE_PATH_TO_PDFs_HERE/
```

Mind that you will need some space on your disk (for local development).
2. If you are dealing with older documents you might need to perform the Optical Character Recognition (OCR) before extracting the content. There are many solutions to perform the OCR. We used the [Acrobat DC](https://acrobat.adobe.com/us/en/acrobat/how-to/ocr-software-convert-pdf-to-text.html#:~:text=Open%20a%20PDF%20file%20containing,to%20edit%20and%20start%20typing.) solution for batch processing.  By the way, this works best in [Adobe Creative Cloud](https://www.adobe.com/creativecloud.html?sdid=8DN85NTZ&mv=search&skwcid=AL!3085!3!442333149296!e!!g!!adobe%20creative%20cloud&ef_id=Cj0KCQjw8rT8BRCbARIsALWiOvQddeIOP51iRNhCzZXx5JBmnNJHmryoYf7LcrMwkpwIZWk70D8OsfYaAtP6EALw_wcB:G:s&s_kwcid=AL!3085!3!442333149296!e!!g!!adobe%20creative%20cloud!1463162620!57154868352&gclid=Cj0KCQjw8rT8BRCbARIsALWiOvQddeIOP51iRNhCzZXx5JBmnNJHmryoYf7LcrMwkpwIZWk70D8OsfYaAtP6EALw_wcB) that offers great value for money.

## The process

The process is shown below.
![The process flow](https://raw.githubusercontent.com/SalesAppi/JPSSM/master/Process_figure.png)

## Stage 1
### General preparations
For the users without prior experience in coding, please consider that some packages used in modeling need to be installed in Google Colab before the first use. Please install them one at a time using _pip_ command:
```bash
pip install gensim
pip install nltk
```
### Corpus preparation
First, we need to parse the XML documents in order to extract the text and record the year of publication. We will prepare a script which will compile a data table for modeling.
We will need several libraries:
```python
import pandas as pd
import os
from re import sub
import lxml.etree as et
import string
import time
import re
```
Next we will define basic functions which also pre-process the text (Stage 2, point 1).
```python
def get_article_body(article_file, tag_path_elements=None):

    if tag_path_elements is None:
        tag_path_elements = ("/",
                             "article",
                             "body")

    article_tree = et.parse(article_file)
    article_root = article_tree.getroot()
    tag_location = '/'.join(tag_path_elements)
    body = article_root.xpath(tag_location)
    body_text = et.tostring(body[0], encoding='unicode', method='text')
   
    
    body_text = body_text.strip().replace('  ', ' ')
    body_text = re.sub(r'\b(-)\b', "_", body_text)
    body_text = body_text.replace(u'\xad', "-")
    body_text = body_text.replace(u'\u2212', "-")
    body_text = body_text.replace('-\n','')
    body_text = body_text.replace('\n', ' ').replace('\r', '')
    
    
    # delete digits
    body_text = sub(pattern=r"\d", repl=r" ", string=body_text)
    
    # remove punctuation
    translator = str.maketrans(' ',' ', string.punctuation)
    body_text = body_text.translate(translator)
    body_text = os.linesep.join([s for s in body_text.splitlines() if s])


    return body_text

def get_title(article_file, tag_path_elements=None):

    if tag_path_elements is None:
        tag_path_elements = ("/",
                             "article",
                             "front",
                             "article-meta",
                             "title-group",
                            "article-title")

    article_tree = et.parse(article_file)
    article_root = article_tree.getroot()
    tag_location = '/'.join(tag_path_elements)
    article_title = article_root.xpath(tag_location)
    article_title = et.tostring(article_title[0], encoding='unicode', method='text')
    
    article_title = article_title.strip().replace('  ', '')
    article_title = os.linesep.join([s for s in article_title.splitlines() if s])

    return article_title
```

Depending on the type of documents we analyze, the structure in _tag_path_elements_ might be different. It is a good practice to check the structure of XML before executing the code. We used [XML Viewer](https://www.xmlviewer.org) for this purpose.
Next, we removed the copyright text which differs in decades.
```python
copy_2010="Copyright of Journal of Personal Selling  Sales Management is the property of Taylor  Francis Ltd and its content may not be copied or emailed to multiple sites or posted to a listserv without the copyright holders express written permission However users may print download or email articles for individual use"
copy_2000="Copyright of Journal of Personal Selling  Sales Management is the property of ME Sharpe Inc  The copyright in an individual article may be maintained by the author in certain cases Content may not be copied or emailed to multiple sites or posted to a listserv without the copyright holders express written permission However users may print download or email articles for individual use"
```
Now, for the convenience:
```python
the_ef=input("Enter the start of the decade: 2000 or 2010 :")
try:
    if the_ef=="2000":
        the_ef=cl_2000
    else:
        the_ef=cl_2010
except:
    print("Entered wrong format")
```
and next:
```python
bodies=[]
names=[] #this holds the names of files in the same order as 
for filename in os.listdir(folderpath):
    if filename!=".DS_Store":
        names.append(filename)
        filepath = os.path.join(folderpath, filename)
        with open (filepath, 'rt') as fd:
            b=get_article_body(fd)
# Removing the copy
            b=b[1:(len(b)-the_ef)]
            bodies.append(b)
```
We use EndNote to host our library with articles. This simplifies the next step because the PDF documents take the name of the first author and year of publication. We can now scan the folder with the PDF files and get the filename (for control) and year of publication (yes, we could also parsed it ;).
```python
fnames=[] #this holds the file names in the same order as bodies
titles=[] #this holds the titles in the same order as bodies
#
#
# Important: if retrieving the title fails, the filename is returned so it can be checked.
#
#
for filename in os.listdir(folderpath):
    if filename!=".DS_Store":
        fnames.append(filename)
        filepath = os.path.join(folderpath, filename)
        
        with open (filepath, 'rt') as fd:
            try:
                t=get_title(fd)
                titles.append(t)
            except:
                t=fd
                titles.append(filename)
```
Now, let's create one table.
```python
df = pd.DataFrame(list(zip(titles, fnames, bodies)), 
               columns =['Title', 'Filename', "Body"]) 
```
Let's add the 'Year' column:
```python
year_file = df['Filename'].tolist()

years=[]
            
for i in year_file:     
    if len(re.findall('-', i))>1:
        year_pub = i.split('-')[2]
        year_pub = year_pub[0:4]

    else:
        year_pub = i.split('-')[1]
        year_pub = year_pub[0:4]
    years.append(year_pub) 

df['Year'] = years
```
And save the table:
```python
df.to_csv('df_tab.tsv', sep =  '\t', index=False)
```
## Stage 2

We start off by collecting the packages we will be using.
```python
import gensim
from gensim import corpora, models 

import matplotlib.pyplot as plt
import gensim
import numpy as np
import pandas as pd

from gensim.models import CoherenceModel, LdaModel
from gensim.corpora import Dictionary

import os, re, operator, warnings
warnings.filterwarnings('ignore') 
%matplotlib inline
```

We also need to import the NLTK package. Note that it needs to be downloaded also.
```python
import nltk

# Needed for the 1st time only.  Download will take a moment.
nltk.download()
```

Basic definitions of functions we will be using.
```python
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
```
Let's load the stop word file and the set of documents
```python
temp_file = datapath((os.path.join(swPath, 'JPSSM_project.txt')))
stop_words = create_stopwords(temp_file)

df=pd.read_csv('df_tab.tsv', sep='\t')
# df stores the collection of documents. The example here uses TAB as a separator (hence _tsv_ extension). A common CSV can be used.
```
Remember that the structure of the table with data is the following:
|  Document_id| Title |Body|Year |
|--|--|--|--|
| number of a document |  title of the document|text of the document|year of publishing

Create a list of documents to analyze.
```python
papers_list = df['Body'].tolist()
```
### Tokenization and lemmatization (2)
Now we will tokenize and lemmatize the documents.
```python
from nltk import stem
import unicodedata

lemmatizer = stem.WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

text1 = [] # collection of papers
stopped_tokens = []
lemmatized_tokens = []

for i in papers_list:
    raw = i.lower()
    raw = normalize(raw)
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [i for i in tokens if not i in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(i) for i in stopped_tokens]
    text1.append(lemmatized_tokens)
```
### Collocation (3)
In the next step, we will calculate the word frequency and create bigrams
```python
# Define a frequency variable to store the number of occurrences of the word
from collections import defaultdict
frequency = defaultdict(int)

# Counting the number of occurrences of a word in the frequency variable
for text in text1:
     for token in text:
         frequency[token] += 1

# Build an array of only words above 1 in the frequency variable  ptext1 P means passed
ptext1 = [[token for token in text if frequency[token] > 20] for text in text1]

# Add bigrams (only ones that appear 20 times or more).
bigram1 = gensim.models.Phrases(ptext1, min_count=20)

for idx in range(len(ptext1)):
    for token in bigram1[ptext1[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            ptext1[idx].append(token)
btext1 = [bigram1[line] for line in ptext1]
```
### Creating essential inputs (4)
Create dictionary and corpus.
```python
dictionary1 = corpora.Dictionary(btext1)

### Using Extreme filter, <20 times and >50% cover is omitted.
dictionary1.filter_extremes(no_below=20, no_above=0.5)

corpus1 = [dictionary1.doc2bow(text) for text in btext1]
```
### Stage 3
The part below simulates the runs of LDA model to find minimum coherence.
```
from tqdm import tqdm
#Metrics for Topic Models
start = 15 # PK: we should not go below 13 topics listed on the website for submitting to JPSSM
limit = 25 # attn.  Setting big number, takes long.  Approx. 10 min. with 30, PK: we don't have enough documents to test a bigger collection
step = 1

coherence_vals = []
coherence_vals_v = []

for n_topic in tqdm(range(start, limit, step)):
    ldamodel_1=gensim.models.ldamodel.LdaModel(corpus=corpus1,
                                               num_topics=n_topic,
                                               id2word = dictionary1,
                                               random_state=1234,
                                             #  update_every =0,
                                               passes = 50,
                                             #  iterations = 400,
                                               chunksize = 200,
                                             #  eval_every = None,
                                             #  alpha ="auto",
                                             #  eta="auto",
                                             #  per_word_topics = False
                                              )


    coherence_model_lda_v = gensim.models.CoherenceModel(model=ldamodel_1, texts=btext1, dictionary=dictionary1, coherence='c_v')
    coherence_vals_v.append(coherence_model_lda_v.get_coherence())
    coherence_model_lda = gensim.models.CoherenceModel(model=ldamodel_1, corpus = corpus1, dictionary = dictionary1, coherence='u_mass')
    coherence_vals.append(coherence_model_lda.get_coherence())
```
This process can take 2 to 5 minutes (depending on the size of the collection). Once done, we can bring up the results in a graphical form:
```python
# evaluation
x = range(start, limit, step)

fig, ax1 = plt.subplots(figsize=(12,5))

# coherence
c1 = 'darkturquoise'
ax1.plot(x, coherence_vals, 'o-', color=c1)
ax1.set_xlabel('Num Topics')
ax1.set_ylabel('Coherence (u mass)', color=c1); ax1.tick_params('y', colors=c1)

# Vis
ax1.set_xticks(x)
fig.tight_layout()
plt.show()
```

Now, we can select the appropriate number of topics (highest coherence score):
```python
num_topics = 17 # assign the number using the chart produced above (can use print(max(coherence_vals_v)) to get the highest value).
```    
Evaluate the LDA model:
```python
lda_model=gensim.models.ldamodel.LdaModel(corpus1, num_topics=num_topics, id2word = dictionary1, passes=50, random_state=1234)

ldatopics = [[word for word, prob in topic] for topicid, topic in lda_model.show_topics(formatted=False)]
```
And investigate the topics:
```python
pprint(lda_model.show_topics(num_topics=num_topics, num_words=15, formatted=True))
```
We can display as many words per topic as we wish. In the example above we used 15.

Let us save the final model:
```python
from gensim.test.utils import datapath

temp_file = datapath("/LDAModel")
lda_model.save(temp_file)
```
Now let's prepare a table with all documents and their topics distribution.
```python


def topics_document_to_dataframe(topics_document, num_topics):
    res = pd.DataFrame(columns=range(num_topics))
    for topic_weight in topics_document:
        res.loc[0, topic_weight[0]] = topic_weight[1]
    return res

# define the allocation to topics per document
topics1 = [lda_model[corpus1[i]] for i in range(len(btext1))]

document_topic_lda = \
pd.concat([topics_document_to_dataframe(topics_document, num_topics=num_topics) for topics_document in topics1]) \
  .reset_index(drop=True).fillna(0)
```

We will add more data to our table now: file names, years, and decades.
```python
filenames_list = df['Filename'].tolist()
document_topic_lda['Filename']=filenames_list

# binning Year into Decade
bins = [1980, 1990, 2000, 2010]
names = ['1', '2', '3', '4']

d = dict(enumerate(names, 1))

df['Decade'] = np.vectorize(d.get)(np.digitize(df['Year'], bins))
```

Finally, we can export our table:
```python
document_topic_lda.to_csv('documents_topics.csv', index=False)
```
## Similarities (7)
Now, as we have the model ready, we can classify previously unseen documents. Let's load a sample text document:
```python
test_file = codecs.open(r'/test file.txt', encoding='utf-8', errors='ignore').read()
```
Next, we should apply the pre-processing steps (similarly as above):
```python
body_text = test_file.strip().replace('  ', ' ')
body_text = body_text.replace('\n', ' ').replace('\r', '')
body_text = sub(pattern=r"\d", repl=r" ", string=body_text)
translator = str.maketrans(' ',' ', string.punctuation)
body_text = body_text.translate(translator)
body_text = os.linesep.join([s for s in body_text.splitlines() if s])
```
In the next step, we present the sample text to the model:
```python
raw = body_text.lower()
raw = normalize(raw)
raw = row.split()
vec_bow = original.doc2bow(raw)
vec_lda = lda_model[vec_bow]
```
And display the probability vector:
```python
vec_lda
```
Note that the output is presented as a list of tuples.
