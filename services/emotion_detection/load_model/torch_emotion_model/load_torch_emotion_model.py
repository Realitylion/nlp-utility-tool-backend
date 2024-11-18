import pickle
    
def load_torch_emotion_model():
    model = pickle.load(open('models/torch_emotion_detection/torch_emotion_model.pkl', 'rb'))
    return model