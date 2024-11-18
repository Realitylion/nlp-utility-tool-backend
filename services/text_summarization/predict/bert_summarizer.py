import spacy
nlp = spacy.load("en_core_web_sm")
from collections import Counter

def summarize(text, num_sentences=3):
    doc = nlp(text)
    sentences = list(doc.sents)
    word_freq = Counter(token.text.lower() for token in doc if not token.is_stop and not token.is_punct)
    sentence_scores = {}
    for sent in sentences:
        score = sum(word_freq.get(token.text.lower(), 0) for token in sent)
        sentence_scores[sent] = score
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    summary = " ".join([sent.text for sent in top_sentences])
    return summary