# importing libaries
from flask import Flask, request, jsonify
import torch

# importing the load_torch_emotion_model function
from services.emotion_detection.load_model.torch_emotion_model.load_torch_emotion_model import load_torch_emotion_model
from services.emotion_detection.load_model.bert_emotion_model.load_bert_emotion_model import load_bert_emotion_model

# importing the predict function from the torch_emotion_predict.py file
from services.emotion_detection.predict.torch_emotion_model.torch_emotion_predict import predict_emotion as torch_emotion_predict
from services.emotion_detection.predict.bert_emotion_model.bert_emotion_predict import predict_emotion as bert_emotion_predict

# importing text summarization function
from services.text_summarization.predict.bert_summarizer import summarize

# importing the load function from the load_lstm_sentiment_classifier.py file
from services.sentiment_analysis.load_model.load_lstm_sentiment_classifier import load_lstm_sentiment_classifier

# importing the lstm_predict function from the lstm_sentiment_classifier.py file
from services.sentiment_analysis.predict.lstm_predict import lstm_predict

# importing the load function from the load_lstm_sentiment_classifier.py file
from services.sentiment_analysis.load_model.load_lstm_sentiment_classifier import load_lstm_sentiment_classifier

# importing the lstm_predict function from the lstm_sentiment_classifier.py file
from services.sentiment_analysis.predict.lstm_predict import lstm_predict

#importing keyword extraction function
from services.named_extraction.named_entity import extract_named_entities
from services.noun_extraction.noun_extraction import extract_noun_phrases

# creating an instance of the Flask class
app = Flask(__name__)

# default route for the application
@app.route('/')
def hello_world():
    return 'Hello World'

##################################################

# Emotion Detection Models

##################################################

# define the EmotionRNN class
class EmotionRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(EmotionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        x = self.embedding(x)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # We use the output of the last LSTM cell
        return out

# route for the emotion detection models
@app.route('/predict/emotionDetection', methods=['POST'])
def predict():    
    # get body of the request
    result = request.json
    
    # torch_emotion_model
    if result['model_name'] == 'torch_emotion_model':
        # get the text data from the body
        data = result['data']['text']

        # load the model
        model = load_torch_emotion_model()

        # make predictions
        result = torch_emotion_predict(data, model)

    # bert_emotion_model
    elif result['model_name'] == 'bert_emotion_model':
        print('bert_emotion_model')
        # get the text data from the body
        data = result['data']['text']

        # check if text length is greater than 512
        if len(data) > 512:
            return jsonify({'data': data, 'result': 'Text length is greater than 512'}).status_code(400)
        
        # check if confidence threshold is there in the body
        if 'confidenceThreshold' not in result['data']:
            confidence_threshold = 50
        else:
            # check if the confidence threshold is between 50 and 90
            print("confidenceThreshold", result['data']['confidenceThreshold'])
            if result['data']['confidenceThreshold'] < 50 or result['data']['confidenceThreshold'] > 90:
                return jsonify({'data': data, 'result': 'Confidence threshold is not between 50 and 90'})
            else:
                confidence_threshold = result['data']['confidenceThreshold']

        # load the model
        model, tokenizer = load_bert_emotion_model()

        # make predictions
        result = bert_emotion_predict(data, model, tokenizer, confidence_threshold)
        result = {'emotion': result[0], 'confidence': str(result[1])}

    return jsonify({'data': data, 'result': result})

##################################################

# Sentiment Analysis Models

##################################################

# route for the sentiment analysis models
@app.route('/predict/sentimentAnalysis', methods=['POST'])
def predict_sentiment():
    # get body of the request
    req = request.json

    # get the text data from the body
    data = req['data']['text']

    # get the model name from the body
    model_name = req['model_name']

    # check if the model name is lstm_sentiment_classifier
    if model_name == 'lstm_sentiment_classifier':
        # load the model
        model = load_lstm_sentiment_classifier()

        # make predictions
        result = lstm_predict(model, data)
    else:
        result = 'Model not found'

    return jsonify({'data': data, 'result': result})

# route for the text summarization models
@app.route('/text/textSummarization', methods=['POST'])
def text_summarization():
    # get body of the request
    result = request.json

    # get the text data from the body
    data = result['data']['text']

    # get the number of sentences from the body
    if 'num_sentences' not in result['data']:
        num_sentences = 3
    else:
        num_sentences = result['data']['num_sentences']

    # make predictions
    result = summarize(data, num_sentences)

    return jsonify({'data': data, 'result': result})

@app.route('/named/namedExtraction', methods=['POST'])
def keyword_extraction():
    # get body of the request 
    result = request.json
    data = result['data']['text']

    # make predictions 
    result = extract_named_entities(data)

    return jsonify({'data': data, 'result': result})

@app.route('/noun/nounExtraction', methods=['POST'])
def noun_extraction():
    # get body of the request   
    result = request.json
    data = result['data']['text']

    # make predictions  
    result = extract_noun_phrases(data)

    return jsonify({'data': data, 'result': result})

# main driver function
if __name__ == '__main__':
    app.run()
