from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch

def predict_emotion(texts, model, tokenizer, confidence_threshold):
    # Define the label-to-emotion mapping

    label_to_emotion = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise",
        6: "neutral"
    }

    # Tokenize the input texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)  # Apply softmax to get probabilities
        predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
        confidence_scores = torch.max(probabilities, dim=1).values.cpu().numpy()  # Max probability as confidence score

    # Return neutral if confidence is below threshold
    prediction = predictions[0]
    confidence = confidence_scores[0]*100
    if confidence < confidence_threshold:
        prediction = 6  # Neutral label
        confidence = 1 - confidence  # Invert confidence score
    
    return [label_to_emotion[prediction], confidence]