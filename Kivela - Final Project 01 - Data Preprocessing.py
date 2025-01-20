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

