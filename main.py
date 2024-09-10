import pandas as pd
import numpy as np


import utils.read_data as rd 
import spliting_methods.genetic_split_v2 as genetic_split
import spliting_methods.greedy_split as greedy_split
import spliting_methods.PL_split_v2 as linear_programm_split
import utils.organize_patient_data as org
import utils.separate_data_into_test_and_train as sep


def compute_sets(split,patients):
    train_index = np.where(split == 0)[0]
    test_index = np.where(split == 1)[0]
    train_index = train_index.astype(int)
    test_index = test_index.astype(int)   
    train_patient_ids = [patient for index, patient in enumerate(patients) if index in train_index]
    test_patient_ids = [patient for index, patient in enumerate(patients) if index in test_index]  
    return train_patient_ids, test_patient_ids  


def compute_train_test(df, train_patients, test_patients):
   
    # Split the data into train and test based on patient IDs
    X_train = df[df["patient_id"].isin(train_patients)]
    X_test = df[df["patient_id"].isin(test_patients)]  
    
    # Separate features and labels
    y_train = X_train['label']
    y_test = X_test['label']  
    
    return X_train, y_train, X_test, y_test


def extract_patient_ids_genetic(patients,classes, population_size,nb_generations,elite_size_ratio,plot=False)
	train_patient_ids,test_patient_ids, best_fitness, split_genetic = genetic_split.split_genetic(patients,classes, population_size,nb_generations,elite_size_ratio,plot=False)
	train_patient_ids_genetic, test_patient_ids_genetic  = compute_sets(split_genetic,patients)
	return train_patient_ids_genetic,test_patient_ids_genetic


def extract_patient_ids_greedy(classes):
	best_split , best_score, solution_scores = greedy_split.greedy_split(classes,max_iterations=1000)
	train_patient_ids_greedy = best_split[0]
	test_patient_ids_greedy = best_split[1]
	return train_patient_ids_greedy,train_patient_ids_greedy
	       
  
def main(path,method, population_size = 100,nb_generations = 500,elite_size_ratio = 0.3):
	grouped_data,df = rd.read_data(path, num_labels=2)
	patients,lesions,classes, classes_ratio = org.organize_data(grouped_data)
	if method == "genetic":
		train_patient_ids, test_patient_ids = extract_patient_ids_genetic(patients,classes, population_size,nb_generations,elite_size_ratio,plot=False)
	elif method == "greedy":
		train_patient_ids, test_patient_ids = extract_patient_ids_greedy(classes)
	elif method == "pl":
		train_patient_ids, test_patient_ids = linear_programm.pl_split(path)
		
	 X_train, y_train,X_test,y_test = compute_train_test_data(df,train_patient_ids,test_patient_ids) 
	 return  X_train, y_train,X_test,y_test




