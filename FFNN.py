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
# .
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

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(X_train, y_train):
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train).toarray()  # Convert sparse matrix to dense
    X_train_tensor = torch.tensor(X_train_vectorized, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

    # Parameters
    input_dim = X_train_tensor.shape[1]
    hidden_dim = 100  # You can adjust this
    output_dim = len(np.unique(y_train))  # Assuming binary classification for simplicity
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.01

    # DataLoader
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = FFNN(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    return model, vectorizer

def save_model(model, vectorizer, model_path='model/model.pth', vectorizer_path='model/vectorizer.pkl'):
    model_directory = os.path.dirname(model_path)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    # Saves the PyTorch model
    torch.save(model.state_dict(), model_path)
    # Saves the vectorizer
    joblib.dump(vectorizer, vectorizer_path)

def load_model(model_path='model/model.pth', vectorizer_path='model/vectorizer.pkl'):
    # Loads the vectorizer
    vectorizer = joblib.load(vectorizer_path)
    
    # Determine the correct input dimension
    sample_text = ["Sample text to initialize vector size"]
    sample_vector = vectorizer.transform(sample_text).toarray()
    input_dim = sample_vector.shape[1]  # Correct input dimension based on the vectorizer
    
    # Load the PyTorch model
    hidden_dim = 100  # Adjust if necessary
    output_dim = 2  # Adjust based on your classification task
    model = FFNN(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    
    return model, vectorizer

def main():
    # Load the data
    data = pd.read_csv('data/train.csv')
    X_train, y_train = preprocess_data(data)
    
    # Train the model
    model, vectorizer = train_model(X_train, y_train)
    
    # Save the model and vectorizer
    save_model(model, vectorizer)
    
    # Load the model and vectorizer
    model, vectorizer = load_model()
    
    # Load the test data
    test_data = pd.read_csv('data/test.csv')
    X_test, y_test = preprocess_data(test_data)
    
    # Vectorize the test data using the loaded vectorizer
    X_test_vectorized = vectorizer.transform(X_test).toarray()
    X_test_tensor = torch.tensor(X_test_vectorized, dtype=torch.float32)
    
    # Classify test data
    with torch.no_grad():  # Ensure no gradients are computed during inference
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
    
    # Print classification report
    print(classification_report(y_test, predicted.numpy()))

if __name__ == '__main__':
    main()
