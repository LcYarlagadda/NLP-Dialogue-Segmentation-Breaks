import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import ast
import joblib
import os

# Specify the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_text(utterance):
    # Extracts the text from the utterance
    utterance_dict = ast.literal_eval(utterance)
    return utterance_dict['text']

def preprocess_data(data):
    # Preprocesses the data
    data['utterance1_text'] = data['utterance1'].apply(extract_text)
    data['utterance2_text'] = data['utterance2'].apply(extract_text)
    data['combined_text'] = data['utterance1_text'] + " " + data['utterance2_text']
    return data['combined_text'], data['label']

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        x = self.relu(last_output)
        x = self.fc(x)
        return x

def train_model(X_train, y_train):
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
    X_train_tensor = torch.tensor(X_train_vectorized, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)

    input_dim = X_train_tensor.shape[1]
    hidden_dim = 100
    output_dim = len(np.unique(y_train))
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.01

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMModel(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))  # Add an extra dimension for LSTM (batch_size, seq_len, input_dim)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    return model, vectorizer

def main():
    data = pd.read_csv('data/train.csv')
    X_train, y_train = preprocess_data(data)
    model, vectorizer = train_model(X_train, y_train)

    model_path = 'model/model.pth'
    vectorizer_path = 'model/vectorizer.pkl'
    save_model(model, vectorizer, model_path, vectorizer_path)

    model, vectorizer = load_model(model_path, vectorizer_path)

    test_data = pd.read_csv('data/test.csv')
    X_test, y_test = preprocess_data(test_data)
    X_test_vectorized = vectorizer.transform(X_test).toarray()
    X_test_tensor = torch.tensor(X_test_vectorized, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_test_tensor.unsqueeze(1))
        _, predicted = torch.max(outputs, 1)

    print(classification_report(y_test, predicted.cpu().numpy()))

def save_model(model, vectorizer, model_path='model/model.pth', vectorizer_path='model/vectorizer.pkl'):
    model_directory = os.path.dirname(model_path)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    torch.save(model.state_dict(), model_path)
    joblib.dump(vectorizer, vectorizer_path)

def load_model(model_path='model/model.pth', vectorizer_path='model/vectorizer.pkl'):
    vectorizer = joblib.load(vectorizer_path)

    sample_text = ["Sample text to initialize vector size"]
    sample_vector = vectorizer.transform(sample_text).toarray()
    input_dim = sample_vector.shape[1]

    hidden_dim = 100
    output_dim = 2
    model = LSTMModel(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, vectorizer

if __name__ == '__main__':
    main()
