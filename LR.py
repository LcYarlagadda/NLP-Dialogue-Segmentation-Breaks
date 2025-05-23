import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import ast
import joblib  # For saving and loading the model
import os

# 0v0

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


def train_model(X_train, y_train):
    # Trains the model
    #vectorizer = TfidfVectorizer(max_features=10000,)
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    model = LogisticRegression(random_state=0)
    model.fit(X_train_vectorized, y_train)
    return model, vectorizer

def save_model(model, vectorizer, model_path='model/model.pkl', vectorizer_path='model/vectorizer.pkl'):
    model_directory = os.path.dirname(model_path)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    # Saves the model and vectorizer
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

def load_model(model_path = 'model/model.pkl', vectorizer_path = 'model/vectorizer.pkl'):
    # Loads the model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def main():
    # Load the data
    data = pd.read_csv('data/train.csv')
    X_train, y_train = preprocess_data(data)
    
    # Train the model
    model, vectorizer = train_model(X_train, y_train)
    # Save the model
    save_model(model, vectorizer)
    
    # Load the model
    model, vectorizer = load_model()
    
    # Load the test data
    test_data = pd.read_csv('data/test.csv')
    X_test, y_test = preprocess_data(test_data)
    
    # Vectorize the test data
    X_test_vectorized = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vectorized)
    print(classification_report(y_test, predictions))
    
if __name__ == '__main__': 
    main()
