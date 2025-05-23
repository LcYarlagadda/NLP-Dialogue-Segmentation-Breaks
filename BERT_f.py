import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import os

# Specify the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_text(utterance):
    # Extracts the text, user, and intent from the utterance
    utterance_dict = eval(utterance)  # Use eval instead of ast.literal_eval for dataframe apply
    # Concatenate user, intent, and text into a single string
    return f"{utterance_dict['user']} {utterance_dict['intent']} {utterance_dict['text']}"

def preprocess_data(data):
    # Preprocesses the data
    data['utterance1_text'] = data['utterance1'].apply(extract_text)
    data['utterance2_text'] = data['utterance2'].apply(extract_text)
    # Include the category text along with utterance1 and utterance2 texts
    data['combined_text'] = data['utterance1_text'] + " " + data['utterance2_text'] # no cat
    return data['combined_text'], data['label']

def train_model(X_train, y_train):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    X_train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt')
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

    input_ids = X_train_encodings['input_ids'].to(device)
    attention_mask = X_train_encodings['attention_mask'].to(device)

    output_dim = len(np.unique(y_train))
    batch_size = 32
    num_epochs = 10
    learning_rate = 5e-5

    dataset = TensorDataset(input_ids, attention_mask, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=output_dim).to(device)
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
    #data = pd.read_csv('data/train.csv')
    data = pd.read_csv('data/doubled_train.csv')
    X_train, y_train = preprocess_data(data)

    model, tokenizer = train_model(X_train, y_train)

    model_path = 'model/bert_model'
    tokenizer_path = 'model/bert_tokenizer'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)

    model, tokenizer = load_model(model_path, tokenizer_path)

    test_data = pd.read_csv('data/test.csv')
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
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

if __name__ == '__main__':
    main()
