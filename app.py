# Importing necessary libraries
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import streamlit as st

# Load datasets
train_data = pd.read_excel('train_product_data.xlsx')
test_data = pd.read_excel('test_data.xlsx')
test_results = pd.read_csv('test_results.csv')

# Explore and preprocess the data
# For simplicity, we'll only focus on the 'description' and 'product_category_tree' columns
train_data = train_data[['description', 'product_category_tree']]
train_data.dropna(inplace=True)  # Remove rows with missing values

# Clean text data
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)     # Remove extra whitespaces
    return text.strip().lower()          # Convert to lowercase and strip whitespace

train_data['cleaned_description'] = train_data['description'].apply(clean_text)

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(train_data['cleaned_description'],
                                                      train_data['product_category_tree'],
                                                      test_size=0.2,
                                                      random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_valid_tfidf)
accuracy = accuracy_score(y_valid, y_pred)
report = classification_report(y_valid, y_pred)

# Save the model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Streamlit app
st.title('Ecommerce Product Categorization')

# Function to predict categories
def predict_category(description):
    cleaned_description = clean_text(description)
    description_tfidf = vectorizer.transform([cleaned_description])
    prediction = model.predict(description_tfidf)
    return prediction[0]

# User input
description = st.text_area('Enter the product description here')

# Button to predict category
if st.button('Predict Category'):
    predicted_category = predict_category(description)
    st.write(f'Predicted Category: {predicted_category}')