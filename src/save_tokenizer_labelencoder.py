import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder
class SimpleTokenizer:
    def __init__(self, max_words=10000):
        self.max_words = max_words
        self.word2idx = {'<PAD>': 0, '<OOV>': 1}
        self.idx2word = {0: '<PAD>', 1: '<OOV>'}
        self.word_counts = {}
        self.fitted = False
    def fit(self, texts):
        for text in texts:
            for word in text.lower().split():
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, _) in enumerate(sorted_words[:self.max_words-2], start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.fitted = True
    def encode(self, text, max_length=200, truncation=True, padding='max_length'):
        tokens = [self.word2idx.get(word, 1) for word in text.lower().split()]
        if truncation:
            tokens = tokens[:max_length]
        if padding == 'max_length':
            tokens += [0] * (max_length - len(tokens))
        return tokens
csv_path = os.path.join('..', 'data', 'dados_limpos.csv')
df = pd.read_csv(csv_path)
texts = df['text'].astype(str).tolist()
labels = df['label'].astype(str).tolist()
tokenizer = SimpleTokenizer(max_words=10000)
tokenizer.fit(texts)
label_encoder = LabelEncoder()
label_encoder.fit(labels)
os.makedirs('models', exist_ok=True)
with open('models/lstm_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print('Tokenizer e Label Encoder salvos em models/') 