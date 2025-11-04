import re
import os
from collections import Counter

BASIC_STOPWORDS = {
    "yang","dan","di","ke","dari","untuk","pada","atau","dengan","sebagai",
    "the","is","are","a","an","of","to","in","for","on","at","and","or"
}

# Set True bila ingin stemming Bahasa Indonesia (butuh Sastrawi)
USE_ID_STEMMER = True

def _id_stemmer():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        return StemmerFactory().create_stemmer()
    except Exception:
        return None

_ID_STEMMER = _id_stemmer()

def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\d+(\.\d+)?', ' <num> ', text)
    text = re.sub(r'[^a-z\<\> ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str):
    return text.split()

def remove_stopwords(tokens):
    return [t for t in tokens if t not in BASIC_STOPWORDS and len(t) > 1]

def stem(tokens):
    if USE_ID_STEMMER and _ID_STEMMER is not None:
        return _ID_STEMMER.stem(' '.join(tokens)).split()
    else:
        try:
            from nltk.stem import PorterStemmer
            ps = PorterStemmer()
            return [ps.stem(t) for t in tokens]
        except Exception:
            return tokens

def preprocess_text(text: str):
    t = clean(text)
    toks = tokenize(t)
    toks = remove_stopwords(toks)
    toks = stem(toks)
    return toks

def preprocess_folder(input_dir, output_dir, log=True):
    os.makedirs(output_dir, exist_ok=True)
    report = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith('.txt'):
            continue
        raw = open(os.path.join(input_dir, fname), 'r', encoding='utf-8', errors='ignore').read()
        toks = preprocess_text(raw)
        with open(os.path.join(output_dir, fname), 'w', encoding='utf-8') as f:
            f.write(' '.join(toks))
        if log:
            cnt = Counter(toks)
            report.append((fname, len(toks), cnt.most_common(10)))
    return report
