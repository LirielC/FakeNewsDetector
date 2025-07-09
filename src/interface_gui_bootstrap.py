import ttkbootstrap as tb
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import scrolledtext, messagebox
import torch
import torch.nn as nn
import pickle
import os
from datetime import datetime
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
class FakeNewsDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Fake News (Dashboard UI)")
        self.root.geometry("900x700")
        self.root.resizable(False, False)
        self.load_model()
        self.create_widgets()
        self.history = []
    def load_model(self):
        MODEL_PATH = os.path.join('models', 'best_lstm_model.pth')
        TOKENIZER_PATH = os.path.join('models', 'lstm_tokenizer.pkl')
        LABEL_ENCODER_PATH = os.path.join('models', 'label_encoder.pkl')
        with open(TOKENIZER_PATH, 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            self.label_encoder = pickle.load(f)
        vocab_size = len(self.tokenizer.word2idx)
        embedding_dim = 100
        hidden_dim = 128
        num_layers = 2
        dropout = 0.3
        self.model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        self.model.eval()
    def create_widgets(self):
        main_frame = tb.Frame(self.root, bootstyle="light", padding=30)
        main_frame.pack(expand=True, fill=BOTH)
        title = tb.Label(main_frame, text="Painel de Análise", font=("Segoe UI", 32, "bold"), bootstyle="dark")
        title.pack(pady=(0, 8))
        subtitle = tb.Label(main_frame, text="Cole o texto de uma notícia para obter uma análise detalhada sobre sua veracidade.", font=("Segoe UI", 15), bootstyle="secondary")
        subtitle.pack(pady=(0, 25))
        card = tb.Frame(main_frame, bootstyle="white", borderwidth=0, relief="flat")
        card.pack(pady=10, padx=10, ipadx=10, ipady=10)
        label = tb.Label(card, text="Conteúdo para Análise", font=("Segoe UI", 13, "bold"), bootstyle="dark")
        label.pack(anchor="w", padx=10, pady=(10, 0))
        self.text_input = scrolledtext.ScrolledText(card, height=7, width=80, font=("Segoe UI", 12), wrap=tk.WORD, borderwidth=0, relief="flat", bg="#f8f9fa")
        self.text_input.pack(padx=10, pady=(8, 18))
        self.text_input.insert(1.0, "Insira o texto completo da notícia aqui...")
        self.text_input.bind("<FocusIn>", self.clear_placeholder)
        self.text_input.bind("<FocusOut>", self.add_placeholder)
        self.placeholder = True
        self.analyze_btn = tb.Button(card, text="Analisar Conteúdo", bootstyle=PRIMARY, width=35, command=self.predict_text)
        self.analyze_btn.pack(pady=(0, 10), anchor="center")
        self.result_frame = tb.Frame(main_frame, bootstyle="light")
        self.result_frame.pack(pady=(10, 0), fill=X)
        self.result_label = tb.Label(self.result_frame, text="", font=("Segoe UI", 18, "bold"))
        self.result_label.pack(pady=(0, 5))
        self.prob_label = tb.Label(self.result_frame, text="", font=("Segoe UI", 13))
        self.prob_label.pack()
        self.error_label = tb.Label(self.result_frame, text="", font=("Segoe UI", 13), bootstyle="danger")
        self.error_label.pack()
        hist_title = tb.Label(main_frame, text="Histórico de Análises", font=("Segoe UI", 14, "bold"), bootstyle="dark")
        hist_title.pack(pady=(30, 0))
        self.history_text = scrolledtext.ScrolledText(main_frame, height=7, width=100, font=("Segoe UI", 10), wrap=tk.WORD, borderwidth=0, relief="flat", bg="#f8f9fa")
        self.history_text.pack(pady=(5, 0))
        self.history_text.config(state='disabled')
    def clear_placeholder(self, event):
        if self.placeholder:
            self.text_input.delete(1.0, tk.END)
            self.placeholder = False
    def add_placeholder(self, event):
        if not self.text_input.get(1.0, tk.END).strip():
            self.text_input.insert(1.0, "Insira o texto completo da notícia aqui...")
            self.placeholder = True
    def predict_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if self.placeholder or not text or text == "Insira o texto completo da notícia aqui...":
            self.show_error("Por favor, insira o texto da notícia para análise.")
            return
        try:
            tokens = self.tokenizer.encode(text, max_length=200)
            input_tensor = torch.tensor([tokens], dtype=torch.long)
            with torch.no_grad():
                output = self.model(input_tensor)
                prob = output.item()
                pred_idx = int(prob > 0.5)
                pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
            if pred_idx == 0:
                resultado = "🛑 FAKE (FALSA)"
                prob_fake = 1 - prob
                prob_true = prob
                color = "danger"
            else:
                resultado = "✅ TRUE (VERDADEIRA)"
                prob_fake = 1 - prob
                prob_true = prob
                color = "success"
            self.show_result(resultado, prob_fake, prob_true, color)
            timestamp = datetime.now().strftime("%H:%M:%S")
            history_entry = f"[{timestamp}] {resultado} - {text[:60]}...\n"
            self.history.append(history_entry)
            self.history_text.config(state='normal')
            self.history_text.insert(tk.END, history_entry)
            self.history_text.see(tk.END)
            self.history_text.config(state='disabled')
        except Exception as e:
            self.show_error(f"Erro durante a predição: {str(e)}")
    def show_result(self, resultado, prob_fake, prob_true, color):
        self.result_label.config(text=resultado, bootstyle=color)
        self.prob_label.config(text=f"Probabilidade FAKE: {prob_fake:.4f} | Probabilidade TRUE: {prob_true:.4f}", bootstyle=color)
        self.error_label.config(text="")
    def show_error(self, msg):
        self.result_label.config(text="")
        self.prob_label.config(text="")
        self.error_label.config(text=f"Ocorreu um erro\n{msg}")
if __name__ == "__main__":
    app = tb.Window(themename="flatly")
    FakeNewsDashboard(app)
    app.mainloop() 