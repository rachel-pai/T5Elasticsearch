"""
Example script to create elasticsearch documents.
"""
import argparse
import mysql.connector
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import torch
from transformers import T5Tokenizer, T5Model
import os
TOKEN_DIR='/Users/rachelchen/Documents/workplace/elastic/models/tokenizer'
MODEL_DIR='/Users/rachelchen/Documents/workplace/elastic/models/model'
MODEL_NAME = os.environ['MODEL_NAME']

def get_emb(inputs_list,model_name,max_length=512):
    if 't5' in model_name:
        tokenizer = T5Tokenizer.from_pretrained(TOKEN_DIR)
        model = T5Model.from_pretrained(MODEL_DIR)
        inputs = tokenizer.batch_encode_plus(inputs_list, max_length=max_length, pad_to_max_length=True,return_tensors="pt")
        outputs = model(input_ids=inputs['input_ids'], decoder_input_ids=inputs['input_ids'])
        last_hidden_states = torch.mean(outputs[0], dim=1)
        return last_hidden_states.tolist()

    elif 'bert' in model_name:
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        batch_encoding = tokenizer.batch_encode_plus(inputs_list, max_length=max_length, pad_to_max_length=True)

        outputs = model(tf.convert_to_tensor(batch_encoding['input_ids'])) # shape: (batch,sequence length, hidden state)
        embeddings = tf.reduce_mean(outputs[0],1)
        return embeddings.numpy().tolist()

def create_document(doc, emb, index_name):
    return {
        '_op_type': 'index',
        '_index': index_name,
        'text': doc['text'],
        'title': doc['title'],
        'text_vector': emb
    }

def load_dataset_from_mysql():
    # while True:
    #     try:
    #         mydb = mysql.connector.connect(user='root', password='root', host='myapp_db', database='docdb',port=3306)
    #     except mysql.connector.errors.DatabaseError:
    #         continue
    #     break
    mydb = mysql.connector.connect(user='otheruser', password='otheruserpass', host='localhost', database='docdb',port=3306)
    mycursor = mydb.cursor()
    mycursor.execute("SELECT title,description FROM docs")
    myresult = mycursor.fetchall()
    mycursor.close()

    docs = []
    for row in myresult:
        doc = {
            'title': row[0],
            'text': row[1]
        }
        docs.append(doc)
    return docs


def bulk_predict(docs, model_name,batch_size=256):
    """Predict bert embeddings."""
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i: i+batch_size]

        embeddings = get_emb(inputs_list=[doc['text'] for doc in batch_docs], model_name = model_name, max_length=512)

        for emb in embeddings:
            yield emb


def load_dataset(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main(args):
    # create docoments
    print("loading data from mysql....")
    docs = load_dataset_from_mysql()
    print("creating documents...")
    with open(args.data, 'w') as f:
        for doc, emb in zip(docs, bulk_predict(docs,model_name=MODEL_NAME)):
            d = create_document(doc, emb, args.index_name)
            f.write(json.dumps(d) + '\n')

    # create index
    print("creating index in elasticsearch...")
    client = Elasticsearch()
    client.indices.delete(index=args.index_name, ignore=[404])
    with open(args.index_file) as index_file:
        source = index_file.read().strip()
        client.indices.create(index=args.index_name, body=source)

    #index documents
    print("index documents...")
    client = Elasticsearch()
    docs = load_dataset(args.data)
    bulk(client, docs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating elasticsearch documents.')
    # parser.add_argument('--model_name', default='t5-small', help='model name, could be '
    # '"t5-small","t5-base","t5-large","t5-3b" and "t5-11b" for t5')
    parser.add_argument('--index_file', default='index.json', help='Elasticsearch index file.')
    parser.add_argument('--index_name', default='docsearch', help='Elasticsearch index name.')
    parser.add_argument('--data', default='documents.jsonl', help='Elasticsearch documents.')
    args = parser.parse_args()
    main(args)
