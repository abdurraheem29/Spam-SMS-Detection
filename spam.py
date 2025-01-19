import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the dataset
# Replace 'filename.csv' with the appropriate file path
file_path = '/Users/YourUsername/Downloads/archive (2).zip'
data = pd.read_csv(file_path, compression='zip', encoding='latin-1')

# Preprocessing the dataset (Assuming columns 'v1' for labels and 'v2' for messages)
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # Encode labels

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# Step 3: Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Train the Support Vector Machine (SVM) model
model = SVC(kernel='linear', C=1.0)
model.fit(X_train_tfidf, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Step 6: Test with custom messages
def classify_message(message):
    vectorized_message = vectorizer.transform([message])
    prediction = model.predict(vectorized_message)
    return "Spam" if prediction[0] == 1 else "Ham"

# Example usage
example_message = "Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/1234 to claim now."
print(f"Message: {example_message}\nClassification: {classify_message(example_message)}")
