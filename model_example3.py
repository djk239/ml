# 
# CLASSIFICATION USING NAIVE BAYES

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score, classification_report

# data containing labels and feature values
data = pd.read_csv("./HandedPickedData.csv")

# Set features to X and labels to Y
X = data.iloc[:,2:]
y = data['Label']

# Set classification variables
testsize = 0.2
randomstate = 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=randomstate)

# Create Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the model
nb_classifier.fit(X_train, y_train)

# Make predictions
predictions = nb_classifier.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, predictions)

# Find our TRUE POSITIVE, FALSE POSITIVE, FALSE NEGATIVE and TRUE NEGATIVE values
tp = np.diag(cm)
fp = np.sum(cm, axis=0) - tp
fn = np.sum(cm, axis=1) - tp
tn = np.sum(cm) - (tp + fp + fn)
print(tp, fp, fn, tn)

# Evaluate our model
accuracy = accuracy_score(y_test, predictions)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
mcc = ((tp * tn) - (fp * fn)) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
report = classification_report(y_test, predictions)


# Display classification variables
print('Classification Variables :')
print('Model - Naive Bayes')
print(f'Test size: {testsize}')
print(f'Random state: {randomstate}')


# Accuracy information
print('Accuracy Results : ')
print(f'Accuracy: {accuracy}')
print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')
print(f'Matthews Correlation Coefficient: {mcc}')
print('Classification Report:\n\n', report)
