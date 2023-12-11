# 
# CLASSIFICATION USING SVM
#

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
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
c = 4

# Set K fold testing variables
kfoldRandomstate = 1
kfoldsplits = 5


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=randomstate)

# Create SVC classifier
svm_classifier = SVC(kernel='poly', C=c)

# Use one vs rest on SVC
ovr_classifier = OneVsRestClassifier(svm_classifier)

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

# Display classification variables
print('Classification Variables :')
print(f'Test size: {testsize}')
print(f'Random state: {randomstate}')
print(f'C: {c}\n')


# Accuracy information
print('Accuracy Results : ')
print(f'Accuracy: {accuracy}')
print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')
print(f'Matthews Correlation Coefficient: {mcc}')
print('Classification Report:\n\n', report)

# Create StratifiedKFold for classification (k-fold validation object)
kfold = StratifiedKFold(n_splits=kfoldsplits, shuffle=True, random_state=kfoldRandomstate)

# Perform k-fold cross-validation
kfold_results = cross_val_score(ovr_classifier, X, y, cv=kfold, scoring=make_scorer(accuracy_score))

# Print cross-validation results
print("Cross-Validation Results:")
print(kfold_results)
print(f"Mean Accuracy: {np.mean(kfold_results)}")
print(f'Using {kfoldsplits} splits and random state {kfoldRandomstate}')



# used for plotting heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ovr_classifier.classes_, yticklabels=ovr_classifier.classes_)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()