# John Ryan Kivela, MA
# MSIS 672
# UMass Boston
# 5/7/2024
"""
##### Randomized Grid Search

Through multiple iterations we arrived at the following hyperparamters:
- 'n_estimators': [75],
- 'min_samples_split': [100]],
- 'max_depth': [4],
- 'min_impurity_decrease':[0.1]    
    
This set of parameters yields a best score of 0.937.
This is an increase from the default model (accuracy = .914).
Overfitting is mitigated by using the ensemble random forest.

###### Cross Validation Results
1. Cross validation accuracy was very stable around 0.95. 
2. The mean accuracy was 0.95
3. The accuracy was relatively stable.

##### Classification Report

1. Accuracy: Very high at 0.95
2. Precision (true positive):
   - Highest for Bad at .96,
   - With Good and Bad at 0.95
3. Recall (sensitvity or true positive rate): 
   - Highest for Bad at 0.97
4. F1 score (0 to 1, balance between metrics): 
   - Highest for Bad at .96
5. Support (number of actual cases of each class in the dataset): 
   - There is very high support for Standard
   - but very low support for Bad

##### Summary   

The accuracy is high at .937 oacross the validation dataset!

The most important features in this model were Oustanding Debt, Non-payment of Minimum Amount, Interest Rate, And NUmber of Delayed Payments.
The model was marginally stronger at predicting Bad, but by only .01.
The model also produces balanced metrics (F1-Score = .95).

In addition, efforts were taken to otimize depth, impurity, and sample split to counteract overfitting. 
Considerations also needed to be made for computing power. With more sophistcated hardware, better results may be possible. 

Reducing the overall complexity of the model may mitigate the risk of overfitting.



"""