import spacy

_spacy_model = None

def get_spacy_model():
    global _spacy_model
    if _spacy_model is None:
        _spacy_model = spacy.load("en_core_web_sm")
    return _spacy_model
