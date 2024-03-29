import flask
from flask import Flask, Response, request, json
from chatbot_cli import *
from static.constants import *
import time

application = Flask(__name__)

def get_prediction(query):
    clean_query = preprocessing.clean_text(query)
    aiml_response = aiml_parser.get_response(clean_query)
    if aiml_response == '':
        return domain_specific(faq_df, clean_query)
    else:
        return aiml_response, AIML, 100


@application.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        req_data = request.get_json()
        query = req_data['query']
        data = {}
        prediction, corpus, confidence = get_prediction(query)
        data['prediction'] = prediction
        data['source'] = corpus
        data['confidence'] = confidence
        data['timestamp'] = int(round(time.time() * 1000))
        resp = flask.Response(json.dumps(data), status=200)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    else:
        return "I'm thinking really hard"

@application.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
        return flask.send_from_directory('.','index.html')
    else:
        return "I'm thinking really hard"

if __name__ == '__main__':
    application.run(host='0.0.0.0')