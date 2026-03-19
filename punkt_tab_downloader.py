import nltk

def ensure_punkt():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading punkt_tab...")
        nltk.download('punkt_tab')
        print("Done.")