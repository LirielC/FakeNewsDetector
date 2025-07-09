import torch
import torch.nn as nn
import pickle
import os
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
class LSTMLoader:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
    def load(self, model_path, tokenizer_path, label_encoder_path):
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        vocab_size = len(self.tokenizer.word2idx)
        embedding_dim = 100
        hidden_dim = 128
        num_layers = 2
        dropout = 0.3
        self.model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
    def predict(self, text):
        tokens = self.tokenizer.encode(text, max_length=200)
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = output.item()
            pred_idx = int(prob > 0.5)
            pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
            return pred_label, prob
if __name__ == "__main__":
    loader = LSTMLoader()
    loader.load('models/best_lstm_model.pth', 'models/lstm_tokenizer.pkl', 'models/label_encoder.pkl')
    texto = input('Digite o texto da notícia para prever:\n')
    label, prob = loader.predict(texto)
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