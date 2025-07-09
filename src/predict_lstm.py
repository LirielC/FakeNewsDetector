import torch
import torch.nn as nn
import pickle
import os
import numpy as np
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
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        output = self.sigmoid(output)
        return output
MODEL_PATH = os.path.join('models', 'best_lstm_model.pth')
TOKENIZER_PATH = os.path.join('models', 'lstm_tokenizer.pkl')
LABEL_ENCODER_PATH = os.path.join('models', 'label_encoder.pkl')
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)
vocab_size = len(tokenizer.word2idx)
embedding_dim = 100
hidden_dim = 128
num_layers = 2
dropout = 0.3
model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()
def predict(text):
    tokens = tokenizer.encode(text, max_length=200)
    input_tensor = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        output = model(input_tensor)
        prob = output.item()
        pred_idx = int(prob > 0.5)
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        return pred_label, prob
if __name__ == '__main__':
    texto_exemplo = input('Digite o texto da notícia para prever (ou cole aqui):\n')
    label, prob = predict(texto_exemplo)
    if label == 0:
        resultado = "FAKE (FALSA)"
        prob_fake = 1 - prob
        prob_true = prob
    else:
        resultado = "TRUE (VERDADEIRA)"
        prob_fake = 1 - prob
        prob_true = prob
    print(f'\nResultado: {resultado}')
    print(f'Probabilidade de ser FAKE: {prob_fake:.4f}')
    print(f'Probabilidade de ser TRUE: {prob_true:.4f}') 