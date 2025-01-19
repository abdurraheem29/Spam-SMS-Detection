
# SMS Spam Classifier

## Overview
This project is a machine learning-based SMS spam classifier. It uses **TF-IDF (Term Frequency-Inverse Document Frequency)** for feature extraction and a **Support Vector Machine (SVM)** for classification. The model predicts whether an SMS message is spam or legitimate (ham).

## Features
- **Data Preprocessing**: Handles and cleans the input dataset.
- **TF-IDF Vectorization**: Converts SMS messages into numerical features suitable for machine learning.
- **SVM Model**: Implements a linear Support Vector Machine to classify messages.
- **Evaluation Metrics**: Provides accuracy, precision, recall, and F1-score for model performance.
- **Custom Message Testing**: Allows users to test the classifier with custom SMS messages.

## Dataset
The dataset used for this project contains labeled SMS messages as either spam or ham. It is loaded from a `.zip` file. The dataset must have the following structure:
- **v1**: The label (`ham` for legitimate, `spam` for spam).
- **v2**: The SMS message text.

## Project Structure
- **sms_spam_classifier.py**: Main Python script for preprocessing, training, and evaluating the model.
- **Dataset**: Download the dataset from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Requirements
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - scikit-learn

Install the required libraries using:
```bash
pip install pandas numpy scikit-learn
```

## Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/sms-spam-classifier.git
   cd sms-spam-classifier
   ```

2. **Prepare the Dataset**:
   - Place your dataset in the specified path (`Downloads` directory by default).
   - Ensure it is in `.zip` format with the expected structure.

3. **Run the Script**:
   ```bash
   python sms_spam_classifier.py
   ```

4. **Custom Message Testing**:
   Modify the `example_message` variable in the script to test with your custom SMS message.

## Results
- The model achieves an accuracy of **97.94%** on the test dataset.
- Example custom message classification:
  - Input: "Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/1234 to claim now."
  - Classification: Spam

## Improvements
- Implement hyperparameter tuning for SVM to enhance performance.
- Address class imbalance using techniques like SMOTE or weighted class handling.
- Experiment with other classifiers like Random Forest or Gradient Boosting.

## Contributing
Feel free to fork this repository and submit pull requests. Suggestions for new features or improvements are welcome!



