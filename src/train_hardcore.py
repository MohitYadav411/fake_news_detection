import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluate import compute_metrics, plot_confusion_matrix

# Determine device (we know it's CPU, but good practice)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_downsample_data(sample_fraction=0.02):
    """
    Loads raw data, merges title+text, and aggressively downsamples it.
    Returns stratified train and test dataframes.
    """
    print(f"Loading raw data and downsampling to {sample_fraction*100}% for Hardcore tier...")
    fake_df = pd.read_csv('data/raw/Fake.csv')
    true_df = pd.read_csv('data/raw/True.csv')
    fake_df['label'] = 1
    true_df['label'] = 0
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Stratified downsample
    _, df_sample = train_test_split(df, test_size=sample_fraction, random_state=42, stratify=df['label'])
    
    df_sample['text'] = df_sample['title'].fillna('') + " " + df_sample['text'].fillna('')
    df_sample = df_sample[['text', 'label']]
    
    # Train/test split on the sample
    train_df, test_df = train_test_split(df_sample, test_size=0.3, random_state=42, stratify=df_sample['label'])
    print(f"Hardcore Train size: {len(train_df)}, Test size: {len(test_df)}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# -------------------------- LSTM --------------------------

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx]).split()
        # Convert to indices
        seq = [self.vocab.get(word, self.vocab['<UNK>']) for word in text]
        # Pad or truncate
        if len(seq) < self.max_len:
            seq = seq + [self.vocab['<PAD>']] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]
            
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=64, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=dropout if dropout > 0 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Take the last hidden state
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        return self.sigmoid(out).squeeze()

def train_lstm(train_df, test_df):
    print("--- Training LSTM ---")
    # Build a simple vocab
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for text in train_df['text']:
        for word in str(text).split():
            if word not in vocab:
                vocab[word] = len(vocab)
                
    train_dataset = TextDataset(train_df['text'].values, train_df['label'].values, vocab)
    test_dataset = TextDataset(test_df['text'].values, test_df['label'].values, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    model = LSTMClassifier(len(vocab)).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 2
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print(f"LSTM Training completed in {time.time() - start_time:.2f} seconds.")
    
    # Save model
    torch.save(model.state_dict(), 'models/lstm.pt')
    
    # Eval
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (outputs >= 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    metrics = compute_metrics(all_labels, all_preds)
    with open('models/lstm_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    fig = plot_confusion_matrix(all_labels, all_preds)
    fig.savefig('models/lstm_confusion_matrix.png')
    
    print(f"LSTM Results: {metrics}")

# -------------------------- BERT --------------------------

class HFDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_hf_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_bert(train_df, test_df):
    print("--- Training BERT ---")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=128)
    
    train_dataset = HFDataset(train_encodings, train_df['label'].tolist())
    test_dataset = HFDataset(test_encodings, test_df['label'].tolist())
    
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_hf_metrics
    )
    
    start_time = time.time()
    trainer.train()
    print(f"BERT Training completed in {time.time() - start_time:.2f} seconds.")
    
    # Save model
    model.save_pretrained('models/bert')
    tokenizer.save_pretrained('models/bert')
    
    # Eval
    eval_result = trainer.evaluate()
    # Extract only our desired metrics
    metrics = {
        'accuracy': eval_result['eval_accuracy'],
        'precision': eval_result['eval_precision'],
        'recall': eval_result['eval_recall'],
        'f1': eval_result['eval_f1']
    }
    
    with open('models/bert_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Get raw predictions for confusion matrix
    raw_preds = trainer.predict(test_dataset)
    preds = raw_preds.predictions.argmax(-1)
    fig = plot_confusion_matrix(test_df['label'].tolist(), preds)
    fig.savefig('models/bert_confusion_matrix.png')
    
    print(f"BERT Results: {metrics}")

if __name__ == "__main__":
    # We use 2% of the total dataset (~900 items) to ensure CPU training finishes quickly
    train_df, test_df = load_and_downsample_data(sample_fraction=0.02)
    train_lstm(train_df, test_df)
    train_bert(train_df, test_df)
    print("Hardcore tier successfully completed.")
