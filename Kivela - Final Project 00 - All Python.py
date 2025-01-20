#!/usr/bin/env python
# coding: utf-8

# John Ryan Kivela, MA
# 
# UMass Boston
# 
# College of Management
# 
# MSIS - 672 Final Project
# 
# Data Preprocessing
# 
# 4/20/24

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer


# In[2]:


## Load data
df = pd.read_csv('output_file.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.dtypes


# In[7]:


# Wrangle Data


# In[8]:


## Columns to drop later
columns_to_drop = ['ID',
                   'Customer_ID',
                   'Occupation',
                   'Credit_History_Age',
                   'Type_of_Loan',
                   'Month',
                   'Age',
                   'Payment_Behaviour'] # This was hard to let go, but I think it could be useful in a future assessment.


# In[9]:


## Convert 'Monthly_Balance' column to float, handle errors by coercing to NaN
df['Monthly_Balance'] = pd.to_numeric(df['Monthly_Balance'], errors='coerce')

## Convert Num of delayed payments to Int64
df["Num_of_Delayed_Payment"] = df["Num_of_Delayed_Payment"].astype('Int64')

# Convert Payment of min amount to boolean
df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].map({'Yes': True, 'No': False})


# In[10]:


# Drop those Columns from above like they're hot
df = df.drop(columns = columns_to_drop)


# In[11]:


# Deal with NA values
## Identify Variable with NaN values
def find_columns_with_nan(df):
    columns_with_nan = [col for col in df.columns if df[col].isna().any()]
    return columns_with_nan

### Identify Variable with NaN values
columns_with_nan = find_columns_with_nan(df)
print("Columns with NaN values:", columns_with_nan)


# In[12]:


# Impute values for NA with numbers
## Columns to impute
columns_to_impute = ['Monthly_Inhand_Salary',
                     'Num_of_Delayed_Payment',
                     'Changed_Credit_Limit',
                     'Num_Credit_Inquiries',
                     'Amount_invested_monthly',
                     'Monthly_Balance'
                     ]

## make a function
def impute_selected_columns(df, columns_to_impute):
    # Use SimpleImputer with strategy='mean'
    imputer = SimpleImputer(strategy='mean')
    
    # Select columns to impute
    df_to_impute = df[columns_to_impute]
    
    # Impute NaN values in selected columns
    df_imputed = pd.DataFrame(imputer.fit_transform(df_to_impute), 
                              columns = columns_to_impute)
    
    # Update original DataFrame with imputed values
    df[columns_to_impute] = df_imputed
    
    # return the data frame
    return df 

## Impute missing values for selected columns
df = impute_selected_columns(df, 
                             columns_to_impute)


# In[13]:


## Categorical variable wrangling

### Replace NaN values in 'Payment_of_Min_Amount' with 'unknown'
df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].fillna("Unknown")


# In[14]:


# Check for NA
columns_with_nan = find_columns_with_nan(df)
print("Columns with NaN values:", columns_with_nan)


# In[15]:


# Covert to Int64
df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].astype('int64')

# Covert to Int64
df['Num_Credit_Inquiries'] = df['Num_Credit_Inquiries'].astype('int64')


# In[16]:


## One-hot encoding
### List of variables to encode
variables_to_encode = ['Payment_of_Min_Amount']

### Perform one-hot encoding for the variables in variables_to_encode
df_hot = pd.get_dummies(df, columns=variables_to_encode)


# In[17]:


df_hot.dtypes


# In[18]:


# Write DataFrame to CSV file
df_hot.to_csv('Kivela - Final Exam PreProcessed.csv', 
              index=False)


# In[1]:

## Decision Tree Classifier

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer


# In[2]:


# Load Data
df_hot = pd.read_csv('Kivela - Final Exam Preprocessed.csv')


# In[3]:


## Define Variables
TargetVariable = 'Credit_Mix'

X = df_hot.drop(columns = TargetVariable)
y = df_hot[TargetVariable]


# In[4]:


## split the preprocessed data into training and validation
train_X, valid_X, train_y, valid_y = train_test_split(X, 
                                                      y, 
                                                      test_size = 0.3, 
                                                      train_size = 0.7, 
                                                      random_state = 373
                                                     )


# In[5]:


# Create a tree model with defaults
clf = DecisionTreeClassifier(criterion="gini", 
                               #splitter="best", 
                               max_depth=None, 
                               min_samples_split=2, 
                               #min_samples_leaf=1, 
                               #min_weight_fraction_leaf=0.0, 
                               #max_features=None, 
                               random_state=373, 
                               #max_leaf_nodes=None, 
                               min_impurity_decrease=0.0, 
                               #class_weight=None, 
                               #ccp_alpha=0.0
                            )


# In[6]:


# Fit (train) the model
clf.fit(X = train_X,
        y = train_y,
        #sample_weight=None,
        #check_input=True
       )


# In[7]:


# Set some names
FeatureNames = list(valid_X.columns)
ClassNames = list(clf.classes_)


# In[8]:


# Cross-validation on the training set
cv_scores = cross_val_score(clf, X, y, cv=5)

print("Cross-validation scores on training set:", cv_scores)
print("Mean CV accuracy on training set:", cv_scores.mean())
# CV accuracy = estimate of how well the model generalizes to new data.


# In[9]:


## Variable/Feature Importance Plot Data Frame
importances = clf.feature_importances_

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


# Initial Parameter grid tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_impurity_decrease':[0, 1],
    'min_samples_split': [2, 5, 10]
}


# In[11]:


# GridsearchCV
gridSearch = GridSearchCV(clf, 
                           param_grid, 
                           cv=5)

# fit gridsearch to training data
gridSearch.fit(X, y)

# Print best parameters and best score
print('Initial score: ', gridSearch.best_score_)
print('Initial parameters: ', gridSearch.best_params_)


# In[12]:


# Updated Parameter grid tuning (wash, rinse, repeat)
param_grid = {
    'criterion': ['entropy'],
    'max_depth': [8],
    'min_impurity_decrease': [.00001],
    'min_samples_split': [200]
}


# In[13]:


# Updated GridsearchCV (wash, rinse, repeat)
gridSearch = GridSearchCV(clf, 
                           param_grid, 
                           cv=5)

# fit gridsearch to training data
gridSearch.fit(X, y)

# Print best parameters and best score
print('Best score: ', gridSearch.best_score_)
print('Best parameters: ', gridSearch.best_params_)


# In[14]:


# Evaluate the final model on validatoipn data
best_clf = gridSearch.best_estimator_
y_pred = best_clf.predict(valid_X)
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


# In[16]:


# Plot decision tree

# Set some names
FeatureNames = list(valid_X.columns)
ClassNames = list(clf.classes_)

plt.figure(figsize = (12, 8))
plot_tree(best_clf, 
          feature_names = FeatureNames, 
          class_names = ClassNames, 
          filled = True)
plt.show()

# In[1]:

## Random Forest Classifier

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

# In[1]:

## Gradient Boosted Classifier

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import xgboost as xgb


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# In[2]:


# Load Data
df_hot = pd.read_csv('Kivela - Final Exam Preprocessed.csv')


# In[3]:


# Define Variables
TargetVariable = 'Credit_Mix'

X = df_hot.drop(columns = TargetVariable)
y = df_hot[TargetVariable]


# In[4]:


# split the data into training and validation
train_X, valid_X, train_y, valid_y = train_test_split(X, 
                                                      y, 
                                                      test_size = 0.3, 
                                                      train_size = 0.7, 
                                                      random_state = 373,
                                                      # stratify = y
                                                     )


# In[5]:


# Create a boosted tree model
clf_xgb = GradientBoostingClassifier(
                                     #loss="log_loss", 
                                     #learning_rate=0.1, 
                                     n_estimators = 10, 
                                     #subsample=1.0, 
                                     #criterion="friedman_mse", 
                                     #min_samples_split=2, 
                                     #min_samples_leaf=1, 
                                     #min_weight_fraction_leaf=0.0, 
                                     #max_depth=3, 
                                     #min_impurity_decrease=0.0, 
                                     #init=None, 
                                     random_state = 373, 
                                     #max_features=None, 
                                     verbose = 0, 
                                     #max_leaf_nodes=None, 
                                     #warm_start=False, 
                                     #validation_fraction=0.1, 
                                     n_iter_no_change = 10, 
                                     #tol=1e-4, 
                                     #ccp_alpha=0.0
                                     )


# In[6]:


# Fit (train) the model
clf_xgb.fit(X = train_X,
            y = train_y,
            #sample_weight=None,
            #check_input=True
            )


# In[7]:


# Set some names
FeatureNames = list(valid_X.columns)
ClassNames = list(clf_xgb.classes_)


# In[8]:


# Cross-validation on the training set
cv_scores = cross_val_score(clf_xgb, X, y, cv=5)

print("Cross-validation scores on training set:", cv_scores)
print("Mean CV accuracy on training set:", cv_scores.mean())
# CV accuracy = estimate of how well the model generalizes to new data.


# In[9]:


## Variable/Feature Importance Plot Data Frame
importances = clf_xgb.feature_importances_

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


# Hyperparameter optimization

# Initial Parameter grid tuning
param_grid = {
    'n_estimators': [0, 10],
    'min_samples_split': [5, 10],
    'max_depth': [3, 5, 7],
    'min_impurity_decrease': [0.0, 0.1, 0.2],
    'random_state': [373],
}


# In[11]:


# Trying a randomized search instead of gridsearch cuz it keeps crashing
rand = RandomizedSearchCV(clf_xgb,
                         param_grid,
                         n_iter = 1,
                         cv = 3,
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
    'n_estimators': [75],
    'min_samples_split': [100],
    'max_depth': [4],
    'min_impurity_decrease': [0.1],
    'random_state': [373],
}


# In[13]:


# Cross Validation (Wash, Rinse, Repeat)
rand = RandomizedSearchCV(clf_xgb,
                         param_grid,
                         n_iter = 3,
                         cv = 3,
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
y_pred = best_clf.predict(valid_X)
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
plt.xticks([])
plt.yticks([])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()

# Store confusion matrix in variable conf_mat
conf_mat = plt

plt.show()

# In[1]:

## Gradient Boosted Classifier

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import matplotlib.pyplot as plt


# In[2]:


# Load the data
df_hot = pd.read_csv("Kivela - Final Exam PreProcessed.csv")


# In[3]:


# Convert categorical variables to numerical
le = LabelEncoder()
df_hot['Credit_Mix'] = le.fit_transform(df_hot['Credit_Mix'])

# Split the data into features (X) and target variable (y)
X = df_hot.drop(columns=['Credit_Mix'])  # Features
y = df_hot['Credit_Mix']  # Target variable


# In[4]:


# Split data into training and testing sets
train_X, valid_X, train_y, valid_y = train_test_split(X, 
                                                      y, 
                                                      test_size=0.2, 
                                                      random_state=42)


# In[5]:


# Standardize features by scaling
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
valid_X_scaled = scaler.transform(valid_X)


# In[6]:


# Initialize/instantiate MLPClassifier
clf_mlp = MLPClassifier(hidden_layer_sizes=(100,), 
                        activation='relu', 
                        solver='adam', 
                        max_iter=1000,
                       )


# In[7]:


# Train the model
clf_mlp.fit(X = train_X_scaled, 
            y = train_y)


# In[8]:


# Set some names
FeatureNames = list(valid_X.columns)
ClassNames = list(clf_mlp.classes_)


# In[9]:


# Number of layers and nodes
num_layers = clf_mlp.n_layers_
num_nodes = clf_mlp.hidden_layer_sizes

print("Number of Layers:", num_layers)
print("Number of Nodes in Each Hidden Layer:", num_nodes)


# In[10]:


# Cross-validation on the training set
cv_scores = cross_val_score(clf_mlp, X, y, cv=5)

print("Cross-validation scores on training set:", cv_scores)
print("Mean CV accuracy on training set:", cv_scores.mean())
# CV accuracy = estimate of how well the model generalizes to new data.


# In[11]:


# Feature importances are not directly assessed in MLP like they are in other classifiers


# In[12]:


# Define hyperparamters:
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,)],  # Number of neurons
    'activation': ['relu', 'tanh', 'logistic'], 
    'solver': ['adam', 'sgd'],  # Optimization
    'learning_rate': ['constant', 'adaptive'], 
}


# In[13]:


# Went with randomized because of compute needs
rand = RandomizedSearchCV(clf_mlp,
                         param_grid,
                         n_iter = 10,
                         cv = 3,
                         scoring = 'accuracy',
                         random_state = 373)

# Fit the randomized cv
rand.fit(X, y)

# Print best parameters and best score
print('Initial score: ', rand.best_score_)
print('Initial parameters: ', rand.best_params_)


# In[14]:


# Update parameters (wash, rinse, repeat)
# Define hyperparamters:
param_grid = {
    'hidden_layer_sizes': [(10, ), (50, ), (100)],  # Number of neurons in the layer
    'activation': ['relu'], 
    'solver': ['adam'],  # Optimization
    'learning_rate': ['constant'], 
}


# In[15]:


# Updated Randomized Search (wash, rinse, repeat)
rand = RandomizedSearchCV(clf_mlp,
                         param_grid,
                         n_iter = 3,
                         cv = 5,
                         scoring = 'accuracy',
                         random_state = 373)

# Fit the randomized cv
rand.fit(X, y)

# Print best parameters and best score
print('Best score: ', rand.best_score_)
print('Best parameters: ', rand.best_params_)


# In[16]:


# Evaluate the final model on validatoipn data
best_clf = rand.best_estimator_

# Predict on validation data
y_pred = best_clf.predict(valid_X)

# Validation Score
valid_accuracy = best_clf.score(valid_X, valid_y)

print("Accuracy on validation set:", valid_accuracy)


# In[17]:


# Calculate evaluation metrics (Classification Report)
accuracy = accuracy_score(valid_y, y_pred)
precision = precision_score(valid_y, y_pred, average='weighted')
recall = recall_score(valid_y, y_pred, average='weighted')
f1 = f1_score(valid_y, y_pred, average='weighted')
conf_matrix = confusion_matrix(valid_y, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", conf_matrix)

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
