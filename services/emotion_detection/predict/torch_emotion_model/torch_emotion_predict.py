from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
import json
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

emotion_to_idx = {'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3, 'love': 4, 'surprise': 5}

# Build vocabulary
with open('resources/torch_emotion_vocabulary.txt', 'r') as f:
    vocab = json.load(f)

# Preprocess the sentence
def preprocess_sentence(sentence):
    tokens = word_tokenize(sentence.lower())
    tokens = [word for word in tokens if word.isalpha()]  # Keep only alphabetic tokens
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return tokens

# emotion to index mapping
def sentence_to_indices(sentence, vocab):
    return [vocab.get(word, vocab['<UNK>']) for word in sentence]

# Padding function for sentences
def pad_sequence(sequence, max_len=20):  # You can adjust the max_len
    return sequence[:max_len] + [vocab['<PAD>']] * (max_len - len(sequence))

def predict_emotion(sentence, model):
    sentence_indices = sentence_to_indices(preprocess_sentence(sentence), vocab)
    sentence_padded = pad_sequence(sentence_indices, max_len=20)
    sentence_tensor = torch.tensor([sentence_padded], dtype=torch.long)

    with torch.no_grad():
        output = model(sentence_tensor)
        _, predicted = torch.max(output, 1)

    return list(emotion_to_idx.keys())[list(emotion_to_idx.values()).index(predicted.item())]