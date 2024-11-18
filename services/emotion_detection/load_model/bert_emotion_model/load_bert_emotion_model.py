import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Load the model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("models/bert_emotion_detection")
tokenizer = DistilBertTokenizer.from_pretrained("models/bert_emotion_detection")

# Set model to evaluation mode
model.eval()

def load_bert_emotion_model():
    return model, tokenizer