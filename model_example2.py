# 
# CLASSIFICATION USING KNN
#

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score, classification_report

# data containing labels and feature values
data = pd.read_csv("./HandedPickedData.csv")

# Set features to X and labels to Y
X = data.iloc[:,2:]
y = data['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=8)

# Use one vs rest on KNN
ovr_classifier = OneVsRestClassifier(knn_classifier)

# Train the model
ovr_classifier.fit(X_train, y_train)

# Make predictions
predictions = ovr_classifier.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, predictions)

# Find our TRUE POSITIVE, FALSE POSITIVE, FALSE NEGATIVE and TRUE NEGATIVE values
tp = np.diag(cm)
fp = np.sum(cm, axis=0) - tp
fn = np.sum(cm, axis=1) - tp
tn = np.sum(cm) - (tp + fp + fn)

# Evaluate our model
accuracy = accuracy_score(y_test, predictions)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
mcc = ((tp * tn) - (fp * fn)) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
report = classification_report(y_test, predictions)

# Accuracy information
print(f'Accuracy: {accuracy}')
print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')
print(f'Matthews Correlation Coefficient: {mcc}')
print('Classification Report:\n', report)

# used for plotting heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ovr_classifier.classes_, yticklabels=ovr_classifier.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Create StratifiedKFold for classification (k-fold validation object)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Perform k-fold cross-validation
kfold_results = cross_val_score(ovr_classifier, X, y, cv=kfold, scoring=make_scorer(accuracy_score))

# Print cross-validation results
print("Cross-Validation Results:")
print(kfold_results)
print(f"Mean Accuracy: {np.mean(kfold_results)}")