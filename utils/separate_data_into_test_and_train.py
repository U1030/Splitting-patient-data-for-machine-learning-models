import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

cols_to_remove = ['label','tumor_evolution',"VOInum"]

def compute_train_test_data(df, train_patients, test_patients, keep_only_cd8_score=True, exclude_cd8 = False, normalize = True):
    # Split the data into train and test based on patient IDs
    X_train = df[df["patient_id"].isin(train_patients)]
    X_test = df[df["patient_id"].isin(test_patients)]  
    
    # Separate features and labels
    y_train = X_train['label']
    y_test = X_test['label']
    
    if keep_only_cd8_score:
        # Keep only columns with "Score CD8" in their names
        X_train = X_train.filter(like="Score CD8")
        X_test = X_test.filter(like="Score CD8")

    elif exclude_cd8:
        # Identify and drop non-numerical columns
        cols_to_remove.append("Score CD8")
        cols_not_numbers = X_train.select_dtypes(exclude=[np.number]).columns
        X_train = X_train.drop(columns=cols_not_numbers)
        X_test = X_test.drop(columns=cols_not_numbers)
        X_train = X_train.drop(columns=cols_to_remove)
        X_test = X_test.drop(columns=cols_to_remove)  
        if normalize :  
            # Normalize numerical features between 0 and 1
            scaler = MinMaxScaler()        
            X_train[X_train.columns] = scaler.fit_transform(X_train)
            X_test[X_test.columns] = scaler.transform(X_test)

    else:
        # Identify and drop non-numerical columns
        cols_not_numbers = X_train.select_dtypes(exclude=[np.number]).columns
        X_train = X_train.drop(columns=cols_not_numbers)
        X_test = X_test.drop(columns=cols_not_numbers)
        X_train = X_train.drop(columns=cols_to_remove)
        X_test = X_test.drop(columns=cols_to_remove) 
        if normalize :         
            # Normalize numerical features between 0 and 1
            scaler = MinMaxScaler()
            X_train[X_train.columns] = scaler.fit_transform(X_train)
            X_test[X_test.columns] = scaler.transform(X_test)

    # Print the columns of the training data
    #print("--------- X train ---------")
    #print(X_train.columns)
    
    return X_train, y_train, X_test, y_test
