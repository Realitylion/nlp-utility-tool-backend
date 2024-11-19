from tensorflow.keras.models import load_model

def load_lstm_sentiment_classifier():
    # Load the saved model
    loaded_model = load_model('models/lstm_sentiment_classifier/sentiment_analysis_model.h5')

    return loaded_model