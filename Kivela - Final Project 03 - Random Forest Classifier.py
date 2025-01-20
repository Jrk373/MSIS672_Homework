#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# In[2]:


# Load the preprocessed data
df_hot = pd.read_csv('Kivela - Final Exam Preprocessed.csv')


# In[3]:


## Define Variables
TargetVariable = 'Credit_Mix'

X = df_hot.drop(columns = TargetVariable)
y = df_hot[TargetVariable]


# In[4]:


## split the data into training and validation
train_X, valid_X, train_y, valid_y = train_test_split(X, 
                                                      y, 
                                                      test_size = 0.3, 
                                                      train_size = 0.7, 
                                                      random_state = 373
                                                     )


# In[5]:


# Create a tree model with defaults
rf = RandomForestClassifier(random_state=373)


# In[6]:


# Fit (train) the model
rf.fit(X = train_X,
        y = train_y,
        #sample_weight=None,
        #check_input=True
       )


# In[7]:


# Set some names
FeatureNames = list(valid_X.columns)
ClassNames = list(rf.classes_)


# In[8]:


# Cross-validation on the training set
cv_scores = cross_val_score(rf, X, y, cv=5)

print("Cross-validation scores on training set:", cv_scores)
print("Mean CV accuracy on training set:", cv_scores.mean())
# Cross Validation accuracy is an estimate of how well the model generalizes to new data.


# In[9]:


# Feature importance assessment
# shows which features are most influential on the model

## Variable/Feature Importance Plot Data Frame
importances = rf.feature_importances_

# Get standard deviation
std = np.std(importances, 
             axis = 0, 
             dtype=None, 
             out=None, 
             ddof=0, 
             keepdims=False,
             # where=None
             )

# Describe Importances
importance_plot_df = pd.DataFrame({'feature': FeatureNames, 
                                   'importance': importances, 
                                   'std': std})

importance_plot_df = importance_plot_df.sort_values('importance')

print(importance_plot_df)

## Importance Plot
ax = importance_plot_df.plot(kind='barh', 
                             xerr='std', 
                             x='feature', 
                             legend=False)
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
ax.set_title('Importance Plot')
plt.tight_layout()
plt.show()


# In[10]:


# Hyperparameter Optimization

# Initial Parameter grid tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15],
    'min_impurity_decrease':[0, 1],
    'min_samples_split': [2, 5, 10]
}


# In[11]:


# Try a randomized search instead of gridsearch, cuz it keeps crashing. Need more compute.
rand = RandomizedSearchCV(rf,
                         param_grid,
                         n_iter = 10,
                         cv = 5,
                         scoring = 'accuracy',
                         random_state = 373)

# Fit the randomized cv
rand.fit(X, y)

# Print best parameters and best score
print('Initial score: ', rand.best_score_)
print('Initial parameters: ', rand.best_params_)


# In[12]:


# Updated Parameter grid tuning (wash, rinse, repeat)
param_grid = {
    #'n_estimators': [300],
    'criterion': ['entropy'],
    'max_depth': [15],
    'min_impurity_decrease':[0.00001],
    'min_samples_split': [2]
}


# In[13]:


# Cross Validation (Wash, Rinse, Repeat)
rand = RandomizedSearchCV(rf,
                         param_grid,
                         n_iter = 10,
                         cv = 5,
                         scoring = 'accuracy',
                         random_state = 373)

# Fit the randomized cv
rand.fit(X, y)

# Print best parameters and best score
print('Best score: ', rand.best_score_)
print('Best parameters: ', rand.best_params_)


# In[14]:


# Evaluate the final model on validatoipn data
best_clf = rand.best_estimator_

# Predict on validation data
y_pred = best_clf.predict(valid_X)

# Validation Score
valid_accuracy = best_clf.score(valid_X, valid_y)

print("Accuracy on validation set:", valid_accuracy)


# In[15]:


# Classification report
print("Classification Report:")
print(classification_report(valid_y, y_pred, target_names=ClassNames))

# Confusion matrix
print("Confusion Matrix:")
conf_mat = confusion_matrix(valid_y, y_pred)
print(conf_mat)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()

# Store confusion matrix in variable conf_mat
conf_mat = plt

plt.show()

