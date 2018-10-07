#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 11:38:53 2018

@author: jai
"""

import tensorflow as tf
from google.cloud import bigquery as bq
import os
from tensorflow.contrib import lookup
from tensorflow.python.platform import gfile
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import json
from googleapiclient import errors
import time
import googleapiclient

#Environment Variables
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/jai/Downloads/NodeOCR-60e87848f2d2.json"

#Using BigQuery to Fetch Data, convert to DF and store as CSV
client = bq.Client(project='nodeocr')

QUERY = (
    'SELECT url, title, score FROM `bigquery-public-data.hacker_news.stories` '
    'WHERE LENGTH(title) > 10 AND score > 10 '
    'LIMIT 10')
query_job = client.query(QUERY)  # API request
rows = query_job.result()  # Waits for query to finish

print(rows)

for row in rows:
    print(row)

query="""
SELECT
  ARRAY_REVERSE(SPLIT(REGEXP_EXTRACT(url, '.*://(.[^/]+)/'), '.'))[OFFSET(1)] AS source,
  COUNT(title) AS num_articles
FROM
  `bigquery-public-data.hacker_news.stories`
WHERE
  REGEXP_CONTAINS(REGEXP_EXTRACT(url, '.*://(.[^/]+)/'), '.com$')
  AND LENGTH(title) > 10
GROUP BY
  source
ORDER BY num_articles DESC
LIMIT 10
"""
df = client.query(query).to_dataframe()

query="""
SELECT source, REGEXP_REPLACE(title, '[^a-zA-Z0-9 $.-]', ' ') AS title FROM
(SELECT
  ARRAY_REVERSE(SPLIT(REGEXP_EXTRACT(url, '.*://(.[^/]+)/'), '.'))[OFFSET(1)] AS source,
  title
FROM
  `bigquery-public-data.hacker_news.stories`
WHERE
  REGEXP_CONTAINS(REGEXP_EXTRACT(url, '.*://(.[^/]+)/'), '.com$')
  AND LENGTH(title) > 10
)
WHERE (source = 'github' OR source = 'nytimes' OR source = 'techcrunch')
"""

df = client.query(query + " LIMIT 10").to_dataframe()

#Splitting the Data
traindf = client.query(query + " AND MOD(ABS(FARM_FINGERPRINT(title)),4) > 0").to_dataframe()
evaldf  = client.query(query + " AND MOD(ABS(FARM_FINGERPRINT(title)),4) = 0").to_dataframe()

traindf['source'].value_counts()
evaldf['source'].value_counts()

traindf.to_csv('train.csv', header=False, index=False, encoding='utf-8', sep='\t')
evaldf.to_csv('eval.csv', header=False, index=False, encoding='utf-8', sep='\t')

!head -3 train.csv

!wc -l *.csv


#Using TensorFlow -- ALREADY IMPLEMENTED IN MODEL.PY
MAX_DOCUMENT_LENGTH = 5  
PADWORD = 'ZYXW'

# vocabulary
lines = ['Some title', 'A longer title', 'An even longer title', 'This is longer than doc length']

# create vocabulary
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
vocab_processor.fit(lines)
with gfile.Open('vocab.tsv', 'wb') as f:
    f.write("{}\n".format(PADWORD))
    for word, index in vocab_processor.vocabulary_._mapping.items():
      f.write("{}\n".format(word))
N_WORDS = len(vocab_processor.vocabulary_)
print ('{} words into vocab.ts'.format(N_WORDS))

# can use the vocabulary to convert words to numbers
table = lookup.index_table_from_file(
  vocabulary_file='vocab.tsv', num_oov_buckets=1, vocab_size=None, default_value=-1)
numbers = table.lookup(tf.constant(lines[0].split()))
with tf.Session() as sess:
  tf.tables_initializer().run()
  print ("{} --> {}".format(lines[0], numbers.eval()))

!cat vocab.tsv

# string operations
titles = tf.constant(lines)
words = tf.string_split(titles)
densewords = tf.sparse_tensor_to_dense(words, default_value=PADWORD)
numbers = table.lookup(densewords)

# now pad out with zeros and then slice to constant length
padding = tf.constant([[0,0],[0,MAX_DOCUMENT_LENGTH]])
padded = tf.pad(numbers, padding)
sliced = tf.slice(padded, [0,0], [-1, MAX_DOCUMENT_LENGTH])

with tf.Session() as sess:
  tf.tables_initializer().run()
  print ("titles=", titles.eval(), titles.shape)
  print ("words=", words.eval())
  print ("dense=", densewords.eval(), densewords.shape)
  print ("numbers=", numbers.eval(), numbers.shape)
  print ("padding=", padding.eval(), padding.shape)
  print ("padded=", padded.eval(), padded.shape)
  print ("sliced=", sliced.eval(), sliced.shape)
  
%%bash  
grep "^def" txtcls1/trainer/model.py

#Train Locally
%%bash
echo "bucket=jai-kotia-text-classification"
rm -rf outputdir
export PYTHONPATH=${PYTHONPATH}:${PWD}/txtcls1
python -m trainer.task \
   --bucket=jai-kotia-text-classification \
   --output_dir=outputdir \
   --job-dir=./tmp --train_steps=200
   
#Train on GCloud   
%%bash
OUTDIR=gs://jai-kotia-text-classification/txtcls1/trained_model
JOBNAME=txtcls_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME
gsutil -m rm -rf $OUTDIR
gsutil cp txtcls1/trainer/*.py $OUTDIR
gcloud ml-engine jobs submit training $JOBNAME \
   --region=asia-east1 \
   --module-name=trainer.task \
   --package-path=/home/jai/Downloads/Datasets/TextClassification/txtcls1/trainer \
   --job-dir=$OUTDIR \
   --staging-bucket=gs://jai-kotia-text-classification \
   --scale-tier=BASIC --runtime-version=1.2 \
   -- \
   --bucket=jai-kotia-text-classification \
   --output_dir=${OUTDIR} \
   --train_steps=36000

#Deploy Model
%%bash
gsutil ls gs://jai-kotia-text-classification/txtcls1/trained_model/export/Servo/

%%bash
MODEL_NAME="txtclass"
MODEL_VERSION="v1"
MODEL_LOCATION=$(gsutil ls gs://jai-kotia-text-classification/txtcls1/trained_model/export/Servo/ | tail -1)
echo "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"
#gcloud ml-engine versions delete ${MODEL_VERSION} --model ${MODEL_NAME}
#gcloud ml-engine models delete ${MODEL_NAME}
gcloud ml-engine models create ${MODEL_NAME} --regions asia-northeast1
gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION}



#API Call

request_data = [
      {
        'title': 'Supreme Court to Hear Major Case on Partisan Districts'
      },
      {
        'title': 'Furan -- build and push Docker images from GitHub to target'
      },
      {
        'title': 'Time Warner will spend $100M on Snapchat original shows and ads'
      },
]

#credentials = GoogleCredentials.get_application_default()
#api = discovery.build('ml', 'v1beta1', credentials=credentials, discoveryServiceUrl='https://storage.googleapis.com/cloud-ml/discovery/ml_v1beta1_discovery.json')

#parent = 'projects/%s/models/%s/versions/%s' % ('nodeocr', 'txtclass', 'v1')
#response = api.projects().predict(body=request_data, name=parent).execute()
#print ("response={0}".format(response))


def predict_json(project, model, instances, version=None):
   
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']

predict_json('nodeocr', 'txtclass', request_data, 'v1')

'''
Out[72]: 
[{'class': 2,
  'prob': [0.020865295082330704, 0.00277658156119287, 0.9763581156730652],
  'source': 'techcrunch'},
 {'class': 2,
  'prob': [0.3123546242713928, 0.01239735260605812, 0.6752480864524841],
  'source': 'techcrunch'},
 {'class': 2,
  'prob': [0.020865295082330704, 0.00277658156119287, 0.9763581156730652],
  'source': 'techcrunch'}]
'''
