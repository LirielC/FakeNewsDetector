import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import random
import matplotlib.pyplot as plt
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=200):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer.encode(text, max_length=self.max_len, truncation=True, padding='max_length')
        return {'input_ids': torch.tensor(tokens, dtype=torch.long), 'labels': torch.tensor(label, dtype=torch.float)}
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
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
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
class LSTMFakeNewsDetector:
    def __init__(self, max_words=10000, max_len=200, embedding_dim=100):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = SimpleTokenizer(max_words)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def load_and_prepare_data(self, file_path):
        df = pd.read_csv(file_path)
        if 'text' not in df.columns or 'label' not in df.columns:
            return None, None
        df = df.dropna(subset=['text', 'label'])
        df['text'] = df['text'].astype(str)
        return df['text'].values, df['label'].values
    def preprocess_text(self, texts):
        self.tokenizer.fit(texts)
        return texts
    def encode_labels(self, labels):
        encoded_labels = self.label_encoder.fit_transform(labels)
        return encoded_labels
    def create_model(self, vocab_size):
        self.model = LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3
        ).to(self.device)
        return self.model
    def train_model(self, train_loader, val_loader, epochs=20, learning_rate=0.001):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(input_ids).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = self.model(input_ids).squeeze()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/best_lstm_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    break
            scheduler.step(avg_val_loss)
        self.model.load_state_dict(torch.load('models/best_lstm_model.pth'))
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
    def evaluate_model(self, test_loader):
        self.model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        criterion = nn.BCELoss()
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids).squeeze()
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                predicted = (outputs > 0.5).float()
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)
        avg_test_loss = test_loss / len(test_loader)
        test_acc = test_correct / test_total
        return avg_test_loss, test_acc
    def predict_text(self, text):
        self.model.eval()
        tokens = self.tokenizer.encode(text, max_length=self.max_len, truncation=True, padding='max_length')
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = output.item()
            pred_idx = int(prob > 0.5)
            pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        return pred_label, prob
    def save_model(self):
        torch.save(self.model.state_dict(), 'models/lstm_model.pth')
        with open('models/lstm_tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        with open('models/lstm_label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
    def load_model_from_file(self):
        self.model = LSTMModel(
            vocab_size=len(self.tokenizer.word2idx),
            embedding_dim=self.embedding_dim,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3
        ).to(self.device)
        self.model.load_state_dict(torch.load('models/lstm_model.pth'))
    def plot_confusion_matrix(self, cm, classes):
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusão')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')
        plt.tight_layout()
    def plot_training_history(self, history):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_losses'], label='Train Loss')
        plt.plot(history['val_losses'], label='Val Loss')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss')
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracies'], label='Train Acc')
        plt.plot(history['val_accuracies'], label='Val Acc')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.legend()
        plt.title('Acurácia')
        plt.tight_layout()
        plt.savefig('results/lstm_training_history.png', dpi=300, bbox_inches='tight')
    def save_training_history(self, history):
        with open('results/lstm_training_history.pkl', 'wb') as f:
            pickle.dump(history, f)
if __name__ == "__main__":
    detector = LSTMFakeNewsDetector()
    X, y = detector.load_and_prepare_data(os.path.join('..', 'data', 'dados_limpos.csv'))
    if X is not None and y is not None:
        detector.preprocess_text(X)
        y_encoded = detector.encode_labels(y)
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y_encoded[:train_size], y_encoded[train_size:]
        train_dataset = TextDataset(X_train, y_train, detector.tokenizer, detector.max_len)
        val_dataset = TextDataset(X_val, y_val, detector.tokenizer, detector.max_len)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        detector.create_model(len(detector.tokenizer.word2idx))
        history = detector.train_model(train_loader, val_loader, epochs=10)
        detector.save_model()
        detector.save_training_history(history)
        detector.plot_training_history(history) 