# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 20:18:36 2023

@author: KMarg
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the adult.data
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
raw_train_df = pd.read_csv(data_url, header=None, names=col_names, sep=', ', engine='python')
train_df = raw_train_df.copy()
train_df.replace("?", np.nan, inplace=True)
train_df.dropna(inplace=True)
train_df['income'] = train_df['income'].apply(lambda x: 1 if x in ('>50K', '>50K.') else 0)
cat_columns = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
categories = {}
for col in cat_columns:
    categories[col] = sorted(set(train_df[col].unique()))
for col in categories:
    train_df[col] = pd.Categorical(train_df[col], categories=categories[col])
train_df = pd.get_dummies(train_df, columns=list(categories.keys()))


# Load and preprocess the adult.test
test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
raw_test_df = pd.read_csv(test_url, header=None, names=col_names, sep=', ', engine='python', skiprows=1)
test_df = raw_test_df.copy()
test_df.replace("?", np.nan, inplace=True)
test_df.dropna(inplace=True)
test_df['income'] = test_df['income'].apply(lambda x: 1 if x in ('>50K', '>50K.') else 0)
for col in categories:
    test_df[col] = pd.Categorical(test_df[col], categories=categories[col])
test_df = pd.get_dummies(test_df, columns=list(categories.keys()))

# Train a Random Forest Classifier
X_train = train_df.drop('income', axis=1)
y_train = train_df['income']
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Test the Random Forest Classifier
X_test = test_df.drop('income', axis=1)
y_test = test_df['income']
y_pred = rfc.predict(X_test)

# Evaluate the model
test_auc = roc_auc_score(y_test, y_pred)
test_fpr, test_tpr, _ = roc_curve(y_test, y_pred)
test_cm = confusion_matrix(y_test, y_pred)

# Print accuracy and classification report for Random Forest Classifier
print("Random Forest model performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", test_cm)

# Plot ROC curve
plt.plot(test_fpr, test_tpr, label=f"Random Forest Classifier (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Plot confusion matrix
sns.heatmap(test_cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - Random Forest Classifier')
plt.show()

# Train a Neural Network
nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
nn.fit(X_train, y_train)

# Test the Neural Network
y_pred = nn.predict(X_test)

# Evaluate the model
test_auc = roc_auc_score(y_test, y_pred)
test_fpr, test_tpr, _ = roc_curve(y_test, y_pred)
test_cm = confusion_matrix(y_test, y_pred)

# Print accuracy and classification report for Neural Network
print("\nNeural Network model performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", test_cm)

# Plot ROC curve
plt.plot(test_fpr, test_tpr, label=f"Neural Network (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Plot confusion matrix
sns.heatmap(test_cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - Neural Network')
plt.show()

# Train decision tree with pruning
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Test decision tree
y_pred = tree.predict(X_test)

# Evaluate model performance
test_auc = roc_auc_score(y_test, y_pred)
test_fpr, test_tpr, _ = roc_curve(y_test, y_pred)
test_cm = confusion_matrix(y_test, y_pred)

print("Decision Tree model performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", test_cm)

#Plot tree
plt.figure(figsize=(20,20))
plot_tree(tree, feature_names=X_train.columns, class_names=["<=50K", ">50K"], filled=True, fontsize=10)
plt.show()

# Plot ROC curve
plt.plot(test_fpr, test_tpr, label=f"Decision Tree (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Plot confusion matrix
sns.heatmap(test_cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - Decision Tree')
plt.show()

# Create KNN model
knn = KNeighborsClassifier()

# Fit model on training data
knn.fit(X_train, y_train)

# Make predictions on test data
y_pred = knn.predict(X_test)

# Evaluate model performance
test_auc = roc_auc_score(y_test, y_pred)
test_fpr, test_tpr, _ = roc_curve(y_test, y_pred)
test_cm = confusion_matrix(y_test, y_pred)

print("KNN model performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", test_cm)

# Plot ROC curve
plt.plot(test_fpr, test_tpr, label=f"KNN (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Plot confusion matrix
sns.heatmap(test_cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - KNN')
plt.show()
