import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
import os

# Specify the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_text(utterance):
    # Extracts the text from the utterance
    utterance_dict = eval(utterance)  # Use eval instead of ast.literal_eval for dataframe apply
    return utterance_dict['text']

def preprocess_data(data):
    # Preprocesses the data
    data['utterance1_text'] = data['utterance1'].apply(extract_text)
    data['utterance2_text'] = data['utterance2'].apply(extract_text)
    data['combined_text'] = data['utterance1_text'] + " " + data['utterance2_text']
    return data['combined_text'], data['label']

# Use RoBERTa
def train_model(X_train, y_train):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=2).to(device)

    model.to(device)
    inputs = tokenizer(list(X_train), padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(y_train.values, dtype=torch.long).to(device)

    batch_size = 8
    num_epochs = 20
    learning_rate = 1e-5

    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    return model, tokenizer

def main():
    data = pd.read_csv('/data/train.csv')
    X_train, y_train = preprocess_data(data)

    model, tokenizer = train_model(X_train, y_train)

    model_path = 'model/disbert_model'
    tokenizer_path = 'model/disbert_tokenizer'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)

    model, tokenizer = load_model(model_path, tokenizer_path)

    test_data = pd.read_csv('/data/test.csv')
    X_test, y_test = preprocess_data(test_data)

    X_test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors='pt')
    input_ids = X_test_encodings['input_ids'].to(device)
    attention_mask = X_test_encodings['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predicted = torch.argmax(outputs.logits, dim=1)

    print(classification_report(y_test, predicted.cpu().numpy()))

def load_model(model_path, tokenizer_path):
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

if __name__ == '__main__':
    main()
