import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, AdamW, AutoTokenizer, AutoModelForSequenceClassification
import os

#adj?...

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
    return data['utterance1_text'], data['utterance2_text'], data['label']

class DualBertForSequenceClassification(nn.Module):
    def __init__(self, model_name_1, model_name_2, num_labels):
        super(DualBertForSequenceClassification, self).__init__()
        self.bert1 = BertModel.from_pretrained(model_name_1)
        self.bert2 = BertModel.from_pretrained(model_name_2)
        self.classifier = nn.Linear(self.bert1.config.hidden_size + self.bert2.config.hidden_size, num_labels)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        outputs1 = self.bert1(input_ids=input_ids1, attention_mask=attention_mask1)
        outputs2 = self.bert2(input_ids=input_ids2, attention_mask=attention_mask2)

        # Concatenate the pooled outputs of both sequences
        pooled_output = torch.cat((outputs1.pooler_output, outputs2.pooler_output), dim=1)
        logits = self.classifier(pooled_output)

        # Wrap logits in a more structured output, mimicking what you might expect from HuggingFace's models
        return {'logits': logits}

def train_model(X_train_1, X_train_2, y_train):
    tokenizer1 = BertTokenizer.from_pretrained('bert-base-cased')
    tokenizer2 = AutoTokenizer.from_pretrained('SpanBERT/spanbert-base-cased')

    # Tokenize both parts
    inputs1 = tokenizer1(list(X_train_1), padding=True, truncation=True, return_tensors='pt')
    inputs2 = tokenizer2(list(X_train_2), padding=True, truncation=True, return_tensors='pt')
    labels = torch.tensor(y_train.values, dtype=torch.long).to(device)

    dataset = TensorDataset(inputs1['input_ids'], inputs1['attention_mask'], inputs2['input_ids'], inputs2['attention_mask'], labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DualBertForSequenceClassification('bert-base-cased', 'SpanBERT/spanbert-base-cased', num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    criterion = nn.CrossEntropyLoss()  # Define the criterion once outside the loop

    model.train()
    for epoch in range(10):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids1, attention_mask1, input_ids2, attention_mask2, labels = [b.to(device) for b in batch]
            output = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            logits = output['logits']  # Extract logits from the model output
            loss = criterion(logits, labels)  # Use extracted logits for the loss calculation
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    return model, tokenizer1, tokenizer2


def main():
    data = pd.read_csv('data/doubled_train.csv')
    X_train_1, X_train_2, y_train = preprocess_data(data)

    model, tokenizer1, tokenizer2 = train_model(X_train_1, X_train_2, y_train)

    test_data = pd.read_csv('data/test.csv')
    X_test_1, X_test_2, y_test = preprocess_data(test_data)

    # Tokenize the test data for both inputs
    inputs_test_1 = tokenizer1(list(X_test_1), padding=True, truncation=True, return_tensors="pt")
    inputs_test_2 = tokenizer2(list(X_test_2), padding=True, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        # Ensure you pass both inputs correctly
        outputs = model(input_ids1=inputs_test_1['input_ids'].to(device),
                        attention_mask1=inputs_test_1['attention_mask'].to(device),
                        input_ids2=inputs_test_2['input_ids'].to(device),
                        attention_mask2=inputs_test_2['attention_mask'].to(device))
        predicted = torch.argmax(outputs['logits'], dim=1)

    print(classification_report(y_test, predicted.cpu().numpy()))

if __name__ == '__main__':
    main()
