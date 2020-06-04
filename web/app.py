from pprint import pprint
from flask import Flask, render_template, jsonify, request
from elasticsearch import Elasticsearch
import os
SEARCH_SIZE = 10
MODEL_NAME = os.environ['MODEL_NAME']
TOKEN_DIR = '/models/tokenizer'
MODEL_DIR = '/models/model'

INDEX_NAME = os.environ['INDEX_NAME']
app = Flask(__name__)
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import torch
from transformers import T5Tokenizer, T5Model

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
        batch_encoding = tokenizer.batch_encode_plus(["this is","the second","the thrid"], max_length=max_length, pad_to_max_length=True)

        outputs = model(tf.convert_to_tensor(batch_encoding['input_ids'])) # shape: (batch,sequence length, hidden state)
        embeddings = tf.reduce_mean(outputs[0],1)
        return embeddings.numpy().tolist()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search')
def analyzer():
    client = Elasticsearch('elasticsearch:9200')

    query = request.args.get('q')
    query_vector = get_emb(inputs_list=[query],model_name =MODEL_NAME,max_length=512)[0]

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc['text_vector']) + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    response = client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": script_query,
            "_source": {"includes": ["title", "text"]}
        }
    )
    print(query)
    pprint(response)
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
