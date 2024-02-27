#!/usr/bin/env python
# coding: utf-8

# # Setup

# ## Import libraries

# In[238]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #, Lasso, Ridge, LassoCV,BayesianRidge
import statsmodels.formula.api as sm
#import matplotlib.pylab as plt

## you need to install dmba library.if you get error message about dmba, 
##please see week 1 "Getting Started with Python" file, 
##Installing dmba in Anaconda Prompt, pip install dmba  

from dmba import regressionSummary#, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import AIC_score #, BIC_score, adjusted_r2_score


# ## Load data

# In[239]:


CHRoster_df_orig = pd.read_csv('CommunityHealthRoster_Demo.csv')
CHRoster_df_orig.head()


# # Method

# ## Select variables for analysis

# In[240]:


selected_columns = ['ProviderName', 
                    'Age', 
                    'Sex', 
                    'Social_Risk_Factor', 
                    'Num_Social_Risk_Factors',
                    'Inpatient_Psych_Service',
                    'Avg_Cost_IP_Facility_Days',
                    'Emergency_Dept_Service',
                    'Alcohol_or_Drug_Diagnosis',
                    'Major_Depressive_Disorder_Diagnosis',
                    #'Race_Black_Indigenous_PeopleOfColor',
                    #'Employed'
                   ]

CHRoster_df = CHRoster_df_orig[selected_columns]

CHRoster_df.head()


# ### Sample based on IP Services

# Select only Inpatient_Psych_Service == 1 becasue we are interested in the effect on IP copst amongst those who actually did have an inpatient stay.

# In[241]:


# Filter for Inpatient_Psych_Service == 1
CHRoster_df = CHRoster_df[CHRoster_df['Inpatient_Psych_Service'] == 1]

CHRoster_df.head()


# #### Remove NaN from Outcome Variable (Avg_Cost_IP_Facility_Days)

# In[242]:


# Count the number of NaN values in 'Avg_Cost_IP_Facility_Days' column
nan_count = CHRoster_df['Avg_Cost_IP_Facility_Days'].isna().sum()

# Display the count of NaN values
print(f"Number of NaN values in 'Avg_Cost_IP_Facility_Days' Before: {nan_count}")

# Remove rows with NaN values in 'Inpatient_Psych_Service' column
CHRoster_df = CHRoster_df.dropna(subset=['Avg_Cost_IP_Facility_Days'])

# Count the number of NaN values in 'Avg_Cost_IP_Facility_Days' column
nan_count = CHRoster_df['Avg_Cost_IP_Facility_Days'].isna().sum()

# Display the count of NaN values
print(f"Number of NaN values in 'Avg_Cost_IP_Facility_Days' After: {nan_count}")


# ## Data Wrangling

# ### Data Types

# In[243]:


CHRoster_df.dtypes


# ### Create age groups

# In[244]:


# Filter cases aged 16 and older
CHRoster_df = CHRoster_df[CHRoster_df['Age'] >= 16]

# Define age groups and labels
age_bins = [16, 25, 35, 50, 65, float('inf')]
age_labels = ['16to25', '26to35', '36to50', '51to65', '65_and_Up']

# Create a new column 'AgeGroup' based on age groups
CHRoster_df['AgeGroup'] = pd.cut(CHRoster_df['Age'], bins=age_bins, labels=age_labels, right=False)

# Convert 'AgeGroup' column to object type
CHRoster_df['AgeGroup'] = CHRoster_df['AgeGroup'].astype('object')

CHRoster_df.dtypes


# ### Create Social Risk Factors groups

# In[245]:


# Count distinct values in 'Num_Social_Risk_Factors' column
distinct_values_count = CHRoster_df['Num_Social_Risk_Factors'].value_counts()

# Display the count of distinct values
print("Distinct Values Count in 'Num_Social_Risk_Factors':")
print(distinct_values_count)


# In[246]:


# Define social risk factor groups and labels
risk_factor_bins = [0, 2, 4, 6, float('inf')]
risk_factor_labels = ['0to2', '3to4', '5to6', '7_and_Up']

# Create a new column 'SocialRiskGroup' based on risk factor groups
CHRoster_df['SocialRiskGroup'] = pd.cut(CHRoster_df['Num_Social_Risk_Factors'], bins=risk_factor_bins, labels=risk_factor_labels, right=False)

CHRoster_df.dtypes


# ## Forge data set

# In[247]:


CHRoster_df.head()


# ### Define and make dummy variables

# In[248]:


# create a list containing predictors' name
predictors = ['Sex',
              'AgeGroup',
              'SocialRiskGroup',
              'Major_Depressive_Disorder_Diagnosis'
             ] 
print(predictors)


# In[249]:


# define outcome/target variable
outcome = 'Avg_Cost_IP_Facility_Days'
print(outcome)


# In[250]:


# check data type of the predictors
#overview of pandas's data type https://pbpython.com/pandas_dtypes.html
CHRoster_df[predictors].dtypes 


# In[251]:


#get k-1 dummies out of k categorical levels by removing the first level
x = pd.get_dummies(CHRoster_df[predictors], drop_first=True)

x.dtypes


# In[252]:


y = CHRoster_df[outcome]
y.head()


# ### Partition Data

# In[253]:


# partition data; split the data training (60%) vs. validation (40%)
# random_state=1: Pass an int for reproducible output across multiple function calls
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4, random_state=1)

train_x.head()


# In[254]:


# check training and validation data sets
data={'Data Set':['train_x', 'valid_x', 'train_y', 'valid_y'], 'Shape':[train_x.shape, valid_x.shape, train_y.shape, valid_y.shape]}
df=pd.DataFrame(data)
df


# ## Modeling

# ### With Pandas

# In[255]:


#build linear regression model using the training data
CHRoster_lm = LinearRegression()
CHRoster_lm.fit(train_x, train_y)


# In[256]:


# print coefficients
print(pd.DataFrame({'Predictor': x.columns, 'Coefficient': CHRoster_lm.coef_}))


# In[257]:


# Get the y intercept
CHRoster_lm.intercept_


# In[259]:


# print performance measures (training data)
regressionSummary(train_y, CHRoster_lm.predict(train_x))


# In[260]:


# Predicted Avg Costs (and Errors) for 20 cases in validation set and summary predictive measures for entire validation set 
# Use predict() to make predictions on a new set
CHRoster_lm_pred = CHRoster_lm.predict(valid_x)
result = pd.DataFrame({'Predicted': CHRoster_lm_pred, 
                       'Actual': valid_y,
                       'Residual': valid_y - CHRoster_lm_pred})
result.head(20)


# In[261]:


# Print performance measures (vaildation data)
regressionSummary(valid_y, CHRoster_lm_pred)


# ### With Statmodels

# In[262]:


train_df = train_x.join(train_y)
train_df.head()


# In[263]:


predictors = train_x.columns
predictors


# In[264]:


# create the linear model formula
#string_name.join(iterable); returns a string concatenated with the elements of iterable
formula = 'Avg_Cost_IP_Facility_Days ~' + '+'.join(predictors)

# Specify the formula for the linear model
# formula = "Avg_Cost_IP_Facility_Days ~ Major_Depressive_Disorder_Diagnosis + Sex_M + AgeGroup_26-35 + AgeGroup_36-50 + AgeGroup_51-65 + AgeGroup_65_and_Up + Q('SocialRiskGroup_3to4') + Q('SocialRiskGroup_5to6') + Q('SocialRiskGroup_7_and_Up')"

formula


# In[265]:


# Build the linear model
CHRoster_sm_lm = sm.ols(formula=formula, data=train_df).fit()

# Display the summary of the linear model
print(CHRoster_sm_lm.summary())


# In[267]:


## check model's accuracy on the training and validation data set
## Training
regressionSummary(train_y, CHRoster_sm_lm.predict (train_x))


# In[268]:


## Validation
CHRoster_sm_lm_pred_stat = CHRoster_sm_lm.predict(valid_x)

# print performance measures (validation data)
regressionSummary(valid_y, CHRoster_sm_lm_pred_stat)


# ### Backward elimination for reducing predictors in CHRoster example

# In[269]:


def train_model(variables):
    model = LinearRegression()
    model.fit(train_x[variables], train_y)
    return model
def score_model(model, variables):
    return AIC_score(train_y, model.predict(train_x[variables]), model)

allVariables = train_x.columns
allVariables


# In[270]:


# backward_elimination is from dmba library provided by the textbook
best_model, best_variables = backward_elimination(allVariables, train_model, score_model, verbose=True)

best_variables


# In[271]:


print(pd.DataFrame({'Predictor': best_variables, 'Coefficient': best_model.coef_}))


# In[272]:


best_model.intercept_


# In[273]:


regressionSummary(valid_y, best_model.predict(valid_x[best_variables]))

