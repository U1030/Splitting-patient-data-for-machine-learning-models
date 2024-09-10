import copy
import random
import matplotlib.pyplot as plt



import utils.read_data as rd 
import utils.organize_patient_data as org

def calculate_total_lesions(dict):
    total_lesions = {0: 0, 1: 0, 2: 0}
    for lesions in dict.values():
        for cls, count in lesions.items():
            total_lesions[cls] += count
    print("total number of lesions in each class : ", total_lesions)
    return total_lesions

def determine_target_counts(total_lesions, train_ratio=2/3):
    train_target = {cls: int(count * train_ratio) for cls, count in total_lesions.items()}
    test_target = {cls: int(total_lesions[cls] - train_target[cls]) for cls in total_lesions}
    print('train target :',train_target)
    print('test target :', test_target)
    return train_target, test_target

def calculate_split_score(train_lesions, test_lesions, train_target, test_target):
    score = 0
    for cls in train_target:
        score += abs(train_lesions[cls] - train_target[cls])
        score += abs(test_lesions[cls] - test_target[cls])
    return score


def greedy_split(patient_dict, train_ratio=2/3, max_iterations=1000):   

    patients = list(patient_dict.keys())
    total_lesions = calculate_total_lesions(patient_dict)
    train_target, test_target = determine_target_counts(total_lesions, train_ratio)

    train_patients = []
    test_patients = patients.copy()
    train_lesions = {0: 0, 1: 0, 2: 0}
    test_lesions = total_lesions.copy()

    solution_scores = []

    # step 1 : add patients to train with greedy approach
    while any(train_lesions[cls] < train_target[cls] for cls in train_target):
        best_patient = None
        best_score = float('inf')
        for patient in test_patients:
            simulated_train_lesions = copy.deepcopy(train_lesions)
            simulated_test_lesions = copy.deepcopy(test_lesions)
            for cls, count in patient_dict[patient].items():
                simulated_train_lesions[cls] += count
                simulated_test_lesions[cls] -= count
            simulated_score = calculate_split_score(simulated_train_lesions, simulated_test_lesions, train_target, test_target)
            if simulated_score < best_score:
                best_score = simulated_score
                best_patient = patient

        if best_patient is not None:
            train_patients.append(best_patient)
            test_patients.remove(best_patient)
            for cls, count in patient_dict[best_patient].items():
                train_lesions[cls] += count
                test_lesions[cls] -= count                
       
    best_split = (copy.deepcopy(train_patients), copy.deepcopy(test_patients))
    best_score = calculate_split_score(train_lesions, test_lesions, train_target, test_target)
    print("step 1 best score :", best_score)
    solution_scores.append(best_score)

    # step 2 : Refinement
    iterations = 0
    best_scores_refinement = []
    while iterations < max_iterations:       
        train_patient = random.choice(train_patients)
        test_patient = random.choice(test_patients)              
       
        new_train_patients = train_patients.copy()
        new_test_patients = test_patients.copy()                                    
        new_train_patients.remove(train_patient)
        new_train_patients.append(test_patient)
        new_test_patients.remove(test_patient)
        new_test_patients.append(train_patient)

        new_train_lesions = {cls: 0 for cls in train_target}
        new_test_lesions = {cls: 0 for cls in test_target}
        for patient in new_train_patients:
            for cls, count in patient_dict[patient].items():
                new_train_lesions[cls] += count
        for patient in new_test_patients:
            for cls, count in patient_dict[patient].items():
                new_test_lesions[cls] += count

        new_score = calculate_split_score(new_train_lesions, new_test_lesions, train_target, test_target)
        if new_score < best_score:
            best_split = (new_train_patients, new_test_patients)
            best_score = new_score                   
            train_patients, test_patients = new_train_patients, new_test_patients                     
            train_lesions, test_lesions = new_train_lesions, new_test_lesions    
            best_scores_refinement.append(best_score) 
            solution_scores.append(best_score)
            print("step 2 : best score :", best_score)
        
        if best_score == 0:
            break
        if len(best_scores_refinement) >= 4 and all(score == best_scores_refinement[-1] for score in best_scores_refinement[-4:]):
            print("Convergence reached. Stopping refinement.")
            break  

        iterations += 1
    
    return best_split, best_score, solution_scores


"""

X_path = "/home/utilisateur/Bureau/CD8_RS_Pipeline/final results/final_data_acses_2_labels.csv"
nb_splits_simulated = 1000
grouped_data,df = rd.read_data(X_path)
patients,lesions,classes, classes_ratio = org.organize_data(grouped_data)

best_split , best_score, solution_scores = greedy_split(classes,max_iterations=1000)
print("best split  train:", best_split[0])
print("best split  test:", best_split[1])
print("best score :", best_score)

x_values = list(range(len(solution_scores)))
plt.plot(x_values, solution_scores, marker='o')
plt.title('Evolution of best score')
plt.xlabel('Iterations')
plt.ylabel('Best score')
plt.show()

nb_lesions_train = sum(value for patient, value in lesions.items() if patient in best_split[0])
print(f'Total number of lesions for patients in train: {nb_lesions_train}')
nb_lesions_test = sum(value for patient, value in lesions.items() if patient in best_split[1])
print(f'Total number of lesions for patients in test: {nb_lesions_test}')

"""
