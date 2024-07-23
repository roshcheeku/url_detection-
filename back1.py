from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Import Scikit-learn helper functions
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Import Scikit-learn models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Import Scikit-learn metric functions
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("\n### Libraries Imported ###\n")

# Load the training data
data_dir = "malicious_phish.csv"
print("- Loading CSV Data -")
df = pd.read_csv(data_dir)

# Drop rows with any NaN values
url_df = df.dropna()

# Perform Train/Test split
test_percentage = 0.2
train_df, test_df = train_test_split(url_df, test_size=test_percentage, random_state=42)
labels = train_df['type']
test_labels = test_df['type']

print("\n### Split Complete ###\n")

# Define tokenizer
def tokenizer(url):
    tokens = re.split('[/-]', url)
    for i in tokens:
        if i.find(".") >= 0:
            dot_split = i.split('.')
            if "com" in dot_split:
                dot_split.remove("com")
            if "www" in dot_split:
                dot_split.remove("www")
            tokens += dot_split
    return tokens

print("\n### Tokenizer defined ###\n")

# Vectorize the training inputs
print("- Training Count Vectorizer -")
cVec = CountVectorizer(tokenizer=tokenizer)
count_X = cVec.fit_transform(train_df['url'])

print("- Training TF-IDF Vectorizer -")
tVec = TfidfVectorizer(tokenizer=tokenizer)
tfidf_X = tVec.fit_transform(train_df['url'])

print("\n### Vectorizing Complete ###\n")

# Vectorize the testing inputs using the same vectorizers
print("- Count Vectorizer -")
test_count_X = cVec.transform(test_df['url'])

print("- TFIDF Vectorizer -")
test_tfidf_X = tVec.transform(test_df['url'])

print("\n### Vectorizing Complete ###\n")

# Define report generator
def generate_report(cmatrix, score, creport):
    plt.figure(figsize=(5, 5))
    sns.heatmap(cmatrix, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues',
                annot_kws={"size": 16}, xticklabels=['bad', 'good'], yticklabels=['bad', 'good'])

    plt.xticks(rotation='horizontal', fontsize=16)
    plt.yticks(rotation='horizontal', fontsize=16)
    plt.xlabel('Actual Label', size=20)
    plt.ylabel('Predicted Label', size=20)

    title = 'Accuracy Score: {0:.4f}'.format(score)
    plt.title(title, size=20)

    print(creport)
    plt.show()

print("\n### Report Generator Defined ###\n")

# Train and evaluate models
models = {
    'Multinomial Naive Bayes (TFIDF)': (MultinomialNB(), tfidf_X, test_tfidf_X),
    'Multinomial Naive Bayes (Count Vectorizer)': (MultinomialNB(), count_X, test_count_X),
    'Logistic Regression (TFIDF)': (LogisticRegression(solver='lbfgs', max_iter=1000), tfidf_X, test_tfidf_X),
    'Logistic Regression (Count Vectorizer)': (LogisticRegression(solver='lbfgs', max_iter=1000), count_X, test_count_X)
}

for model_name, (model, X_train, X_test) in models.items():
    model.fit(X_train, labels)
    score = model.score(X_test, test_labels)
    predictions = model.predict(X_test)
    cmatrix = confusion_matrix(predictions, test_labels)
    creport = classification_report(predictions, test_labels)
    
    print(f"\n### {model_name} Model Built ###\n")
    generate_report(cmatrix, score, creport)

from joblib import dump

# Save models and vectorizers
dump(models['Multinomial Naive Bayes (Count Vectorizer)'][0], 'mnb_count_model.pkl')
dump(models['Multinomial Naive Bayes (TFIDF)'][0], 'mnb_tfidf_model.pkl')
dump(models['Logistic Regression (Count Vectorizer)'][0], 'lgs_count_model.pkl')
dump(models['Logistic Regression (TFIDF)'][0], 'lgs_tfidf_model.pkl')
dump(cVec, 'count_vectorizer.pkl')
dump(tVec, 'tfidf_vectorizer.pkl')
print("Models and vectorizers saved successfully.")

# Define a function to check if a given URL is malicious
def check_url(url, count_vectorizer, tfidf_vectorizer, mnb_count, mnb_tfidf, lgs_count, lgs_tfidf):
    count_vec = count_vectorizer.transform([url])
    tfidf_vec = tfidf_vectorizer.transform([url])

    pred_mnb_count = mnb_count.predict(count_vec)
    print(f"Multinomial Naive Bayes (Count Vectorizer) prediction: {pred_mnb_count[0]}")

    pred_mnb_tfidf = mnb_tfidf.predict(tfidf_vec)
    print(f"Multinomial Naive Bayes (TFIDF) prediction: {pred_mnb_tfidf[0]}")

    pred_lgs_count = lgs_count.predict(count_vec)
    print(f"Logistic Regression (Count Vectorizer) prediction: {pred_lgs_count[0]}")

    pred_lgs_tfidf = lgs_tfidf.predict(tfidf_vec)
    print(f"Logistic Regression (TFIDF) prediction: {pred_lgs_tfidf[0]}")

    predictions = {
        "Multinomial Naive Bayes (Count Vectorizer)": pred_mnb_count[0],
        "Multinomial Naive Bayes (TFIDF)": pred_mnb_tfidf[0],
        "Logistic Regression (Count Vectorizer)": pred_lgs_count[0],
        "Logistic Regression (TFIDF)": pred_lgs_tfidf[0]
    }

    return predictions

def interpret_prediction(prediction):
    if prediction == 'benign':
        return "safe"
    else:
        return "not safe"

print("\n### Ready to check URLs ###\n")

input_url = input("Enter a URL to check: ")
results = check_url(input_url, cVec, tVec, models['Multinomial Naive Bayes (Count Vectorizer)'][0], models['Multinomial Naive Bayes (TFIDF)'][0], models['Logistic Regression (Count Vectorizer)'][0], models['Logistic Regression (TFIDF)'][0])

print("\n### Prediction Results ###\n")
for model, prediction in results.items():
    print(f"{model}: {interpret_prediction(prediction)}")
