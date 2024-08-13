# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 21:02:42 2023

@author: KMarg
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import shap

#Creating a list of column names
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
             'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

# Load the data
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

raw_train_df = pd.read_csv(data_url, header=None, names=col_names, sep=', ', engine='python')

test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

raw_test_df = pd.read_csv(test_url, header=None, names=col_names, sep=', ', engine='python', skiprows=1)

# Combine the train and test sets
combined_df = pd.concat([raw_train_df, raw_test_df], axis=0, ignore_index=True)

# Preprocessing
combined_df.replace("?", np.nan, inplace=True)
combined_df.dropna(inplace=True)
combined_df['income'] = combined_df['income'].apply(lambda x: 1 if x in ('>50K', '>50K.') else 0)

cat_columns = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
categories = {}
for col in cat_columns:
    categories[col] = sorted(set(combined_df[col].unique()))
for col in categories:
    combined_df[col] = pd.Categorical(combined_df[col], categories=categories[col])
combined_df = pd.get_dummies(combined_df, columns=list(categories.keys()))

# Split back into train and test sets with labels
train_df = combined_df.iloc[:raw_train_df.shape[0], :]
test_df = combined_df.iloc[raw_train_df.shape[0]:, :]

X_train, X_test, y_train, y_test = train_test_split(train_df.drop('income', axis=1), train_df['income'], test_size=0.2, random_state=42)
X_train_labels = pd.concat([X_train, y_train], axis=1)
X_test_labels = pd.concat([X_test, y_test], axis=1)


# Train a Random Forest Classifier
X_train = train_df.drop('income', axis=1)
y_train = train_df['income']
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Test the Random Forest Classifier
X_test = test_df.drop('income', axis=1)
y_test = test_df['income']
y_pred = rfc.predict(X_test)

# Train a decision tree with pruning
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Print accuracy and classification report for Random Forest Classifier
print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
rfc_cm = confusion_matrix(y_test, y_pred)
sns.heatmap(rfc_cm, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Random Forest Classifier')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# Train a Neural Network
nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
nn.fit(X_train, y_train)

# Test the Neural Network
y_pred = nn.predict(X_test)

# Print accuracy and classification report for Neural Network
print("\nNeural Network:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
nn_cm = confusion_matrix(y_test, y_pred)
sns.heatmap(nn_cm, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Neural Network')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Evaluate the decision tree on the test set
y_pred = tree.predict(X_test)
print("Decision Tree: ")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
tree_cm = confusion_matrix(y_test, y_pred)
sns.heatmap(tree_cm, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Decision Tree')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

#Train knn clasifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Test the KNN Classifier
y_pred = knn.predict(X_test)

# Print accuracy and classification report for KNN Classifier
print("KNN Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
knn_cm = confusion_matrix(y_test, y_pred)
sns.heatmap(knn_cm, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for KNN Classifier')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# ROC curve for Random Forest Classifier
y_proba_rfc = rfc.predict_proba(X_test)[:, 1]
fpr_rfc, tpr_rfc, _ = roc_curve(y_test, y_proba_rfc)
roc_auc_rfc = auc(fpr_rfc, tpr_rfc)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rfc, tpr_rfc, label='Random Forest Classifier (area = %0.2f)' % roc_auc_rfc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest Classifier')
plt.legend(loc="lower right")
plt.show()

# ROC curve for Neural Network
y_proba_nn = nn.predict_proba(X_test)[:, 1]
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_proba_nn)
roc_auc_nn = auc(fpr_nn, tpr_nn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_nn, tpr_nn, label='Neural Network (area = %0.2f)' % roc_auc_nn)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Neural Network')
plt.legend(loc="lower right")
plt.show()

# ROC curve for Decision Tree
y_proba_tree = tree.predict_proba(X_test)[:, 1]
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_proba_tree)
roc_auc_tree = auc(fpr_tree, tpr_tree)

plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree, label='Decision Tree (area = %0.2f)' % roc_auc_tree)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc="lower right")
plt.show()

# ROC curve for KNN Classifier
y_proba_knn = knn.predict_proba(X_test)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, label='KNN Classifier (area = %0.2f)' % roc_auc_knn)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - KNN Classifier')
plt.legend(loc="lower right")
plt.show()

# All ROC curves together
plt.plot(fpr_rfc, tpr_rfc, label='Random Forest Classifier (area = %0.2f)' % roc_auc_rfc)
plt.plot(fpr_nn, tpr_nn, label='Neural Network (area = %0.2f)' % roc_auc_nn)
plt.plot(fpr_tree, tpr_tree, label='Decision Tree (area = %0.2f)' % roc_auc_tree)
plt.plot(fpr_knn, tpr_knn, label='KNN Classifier (area = %0.2f)' % roc_auc_knn)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

#Plot tree
plt.figure(figsize=(20,20))
plot_tree(tree, feature_names=X_train.columns, class_names=["<=50K", ">50K"], filled=True, fontsize=10)
plt.show()

# define feature names (replace with your own feature names)
feature_names = col_names

# SHAP for Random Forest Classifier
explainer = shap.TreeExplainer(rfc)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test, show=False)
plt.title('SHAP Summary Plot for Random Forest Classifier')
plt.show()

# SHAP for Neural Network
explainer = shap.KernelExplainer(nn.predict_proba, X_train)
shap_values = explainer.shap_values(X_test, nsamples=100)

shap.summary_plot(shap_values[1], X_test, show=False)
plt.title('SHAP Summary Plot for Neural Network')
plt.show()

# SHAP for Decision Tree
explainer = shap.TreeExplainer(tree)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot for Decision Tree')
plt.show()

# SHAP for KNN Classifier
explainer = shap.KernelExplainer(knn.predict_proba, X_train)
shap_values = explainer.shap_values(X_test, nsamples=100)

shap.summary_plot(shap_values[1], X_test, show=False)
plt.title('SHAP Summary Plot for KNN Classifier')
plt.show()