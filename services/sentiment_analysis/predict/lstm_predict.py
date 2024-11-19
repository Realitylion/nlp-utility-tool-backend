from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

def lstm_predict(model, data):
    # Use the loaded model for predictions
    word_index = imdb.get_word_index()
    max_len = 100

    def predict_sentiment_with_loaded_model(review):
        encoded_review = [word_index.get(word, 2) + 3 for word in review.lower().split()]
        padded_review = pad_sequences([encoded_review], maxlen=max_len, padding='post')
        prediction = model.predict(padded_review)[0][0]
        return "Positive" if prediction > 0.5 else "Negative"
    
    return predict_sentiment_with_loaded_model(data)