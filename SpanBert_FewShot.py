import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
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

def train_model(X_train, y_train):
    tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained("SpanBERT/spanbert-base-cased", num_labels=2).to(device)

    inputs = tokenizer(list(X_train), padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(y_train.values, dtype=torch.long).to(device)

    batch_size = 32
    num_epochs = 10
    learning_rate = 5e-5

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
    data = pd.read_csv('data/train.csv')
    X_train, y_train = preprocess_data(data)

    model, tokenizer = train_model(X_train, y_train)

    test_data = pd.read_csv('data/test.csv')
    X_test, y_test = preprocess_data(test_data)

    inputs_test = tokenizer(list(X_test), padding=True, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        inputs_test = {k: v.to(device) for k, v in inputs_test.items()}
        outputs = model(**inputs_test)
        predicted = torch.argmax(outputs.logits, dim=1)

    print(classification_report(y_test, predicted.cpu().numpy()))

if __name__ == '__main__':
    main()
