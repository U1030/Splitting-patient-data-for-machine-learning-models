import pulp
import time
import matplotlib.pyplot as plt


import utils.read_data as rd
import utils.organize_patient_data as org

def calculate_total_lesions(dict):
    total_lesions = {0: 0, 1: 0, 2: 0}
    for lesions in dict.values():
        for cls, count in lesions.items():
            total_lesions[cls] += count
    return total_lesions

def determine_target_counts(total_lesions, train_ratio=2/3):
    train_target = {cls: int(count * train_ratio) for cls, count in total_lesions.items()}
    test_target = {cls: total_lesions[cls] - train_target[cls] for cls in total_lesions}    
    return train_target, test_target

def plot_lesion_distribution(patients_dict, train_patients, test_patients):
    # Initialize dictionaries to store total lesions per label in train and test sets
    train_lesions = {}
    test_lesions = {}    
    # Extract labels from the first patient's data (assuming all patients have the same labels)
    labels = patients_dict[list(patients_dict.keys())[0]].keys()    
    # Initialize counts to zero for each label
    for label in labels:
        train_lesions[label] = 0
        test_lesions[label] = 0    
    # Calculate total lesions per label in the train and test sets
    for patient_id, lesions in patients_dict.items():
        for label, count in lesions.items():
            if patient_id in train_patients:
                train_lesions[label] += count
            elif patient_id in test_patients:
                test_lesions[label] += count    
    # Prepare data for plotting
    labels = list(labels)
    train_counts = [train_lesions[label] for label in labels]
    test_counts = [test_lesions[label] for label in labels]    
    x = range(len(labels))    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35    
    ax.bar(x, train_counts, width, label='Train Patients')
    ax.bar(x, test_counts, width, bottom=train_counts, label='Test Patients')    
    ax.set_xlabel('Lesion Labels')
    ax.set_ylabel('Total Lesions')
    ax.set_title('Lesion Distribution in Train and Test Sets')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    return 

def compute_PL(patient_ids, labels, patients_dict,max_iterations, train_ratio = 2/3, test_ratio = 1/3, alpha = 0,beta = 1):
    # Initialize the problem
    problem = pulp.LpProblem("LesionSplit", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts("x", patient_ids, cat='Binary')
    y = {(i, l): pulp.LpVariable(f"y_{i}_{l}", cat='Binary') for i in patient_ids for l in labels}
    z_train = pulp.LpVariable.dicts("z_train", labels, lowBound=0)
    z_test = pulp.LpVariable.dicts("z_test", labels, lowBound=0)

    # Objective Function
    problem += pulp.lpSum(alpha*z_train[l] + beta*z_test[l] for l in labels), "Minimize deviations"

    # Constraints

    # Ensure that 2/3 of lesions are in the train set and 1/3 are in the test set
    total_lesions = sum(sum(label_count.values()) for label_count in patients_dict.values())
    target_train_lesions =  int((train_ratio) * total_lesions)
    target_test_lesions = int((test_ratio) * total_lesions)
    print('total number of lesions :', total_lesions)
    print('target number of lesions in train :', target_train_lesions)
    print('target number of lesions in test :', target_test_lesions)

    problem += pulp.lpSum(y[(i, l)] * patients_dict[i][l] for i in patient_ids for l in labels) <= target_train_lesions, "Train Lesion Count"
    problem += pulp.lpSum((1 - y[(i, l)]) * patients_dict[i][l] for i in patient_ids for l in labels) >= target_test_lesions, "Test Lesion Count"

    # Ensure that the proportion of each label in the train and test sets is close to 2/3 and 1/3 respectively
    for l in labels:
        total_label_l = sum(patients_dict[i][l] for i in patient_ids)

        # Train label distribution constraints
        problem += z_train[l] >= pulp.lpSum(y[(i, l)] * patients_dict[i][l] for i in patient_ids) - (train_ratio) * total_label_l, f"Train Label {l} Positive Deviation"
        problem += z_train[l] >= -(pulp.lpSum(y[(i, l)] * patients_dict[i][l] for i in patient_ids) - (train_ratio) * total_label_l), f"Train Label {l} Negative Deviation"
        
        # Test label distribution constraints
        problem += z_test[l] >= pulp.lpSum((1 - y[(i, l)]) * patients_dict[i][l] for i in patient_ids) - (test_ratio) * total_label_l, f"Test Label {l} Positive Deviation"
        problem += z_test[l] >= -(pulp.lpSum((1 - y[(i, l)]) * patients_dict[i][l] for i in patient_ids) - (test_ratio) * total_label_l), f"Test Label {l} Negative Deviation"


    # Ensure each patient can only be in one set
    for i in patient_ids:
        for l in labels:
            problem += y[(i, l)] <= x[i], f"Link {i}_{l}"
            problem += y[(i, l)] >= 0, f"Non-negative y_{i}_{l}"

    solver_with_limit = pulp.PULP_CBC_CMD(msg=True, options=[
    'heuristics=on',     # Enable heuristics
    'preprocess=on',     # Enable preprocessing
    'allowableGap=0.1',  # Allowable gap for approximation
    'maxNodes={}'.format(max_iterations),  # Limit on the number of nodes to explore
    'seconds=600'        # Time limit in seconds
    ])

    # Solve the problem
    problem.solve(solver=solver_with_limit)

    # Output results
    train_patients = [i for i in patient_ids if pulp.value(x[i]) == 1]
    test_patients = [i for i in patient_ids if pulp.value(x[i]) == 0]

    print("Train Patients:", train_patients)
    print("Test Patients:", test_patients)

    plot_lesion_distribution(patients_dict, train_patients, test_patients)

    return train_patients, test_patients


def compute_PL_new(patient_ids, labels, patients_dict,max_iterations,target_train, target_test, alpha = 0.2,beta = 0.8):
    print("labels :", labels)

    # Initialize the problem
    problem = pulp.LpProblem("LesionSplit", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts("x", patient_ids, cat='Binary')
    y = {(i, l): pulp.LpVariable(f"y_{i}_{l}", cat='Binary') for i in patient_ids for l in labels}    

    # Objective Function
    
    objective_function = pulp.lpSum(alpha * (pulp.lpSum(y[(i, l)] * patients_dict[i][l] for i in patient_ids) - target_train[l]) +
                                beta * (pulp.lpSum((1 - y[(i, l)]) * patients_dict[i][l] for i in patient_ids) - target_test[l])
                                for l in labels)
    
    problem += objective_function, "Minimize deviations"

    # Constraints
 
    for l in labels: 

        print("---------CLASS---------", l)
        print('----target train ----', target_train[l])  
        print('----target test ----', target_test[l])  

        # Train label distribution constraints
        problem += target_train[l] >= pulp.lpSum(y[(i, l)] * patients_dict[i][l] for i in patient_ids) , f"Train Label {l} Positive Deviation"
        problem += target_train[l] >= -(pulp.lpSum(y[(i, l)] * patients_dict[i][l] for i in patient_ids)) , f"Train Label {l} Negative Deviation"
        
        # Test label distribution constraints
        problem += target_test[l] >= pulp.lpSum((1 - y[(i, l)]) * patients_dict[i][l] for i in patient_ids) , f"Test Label {l} Positive Deviation"
        problem += target_test[l] >= -(pulp.lpSum((1 - y[(i, l)]) * patients_dict[i][l] for i in patient_ids)), f"Test Label {l} Negative Deviation"

    # Ensure each patient can only be in one set
    for i in patient_ids:
        for l in labels:
            problem += y[(i, l)] <= x[i], f"Link {i}_{l}"
            problem += y[(i, l)] >= 0, f"Non-negative y_{i}_{l}"

    solver_with_limit = pulp.PULP_CBC_CMD(msg=True, options=[
    'heuristics=on',     # Enable heuristics
    'preprocess=on',     # Enable preprocessing
    'allowableGap=0.1',  # Allowable gap for approximation
    'maxNodes={}'.format(max_iterations),  # Limit on the number of nodes to explore
    'seconds=600'        # Time limit in seconds
    ])

    # Solve the problem
    problem.solve(solver=solver_with_limit)

    # Output results
    train_patients = [i for i in patient_ids if pulp.value(x[i]) == 1]
    test_patients = [i for i in patient_ids if pulp.value(x[i]) == 0]

    print("Train Patients:", train_patients)
    print("Test Patients:", test_patients)

    plot_lesion_distribution(patients_dict, train_patients, test_patients)

    return train_patients, test_patients


def pl_split(path):
    
    grouped_data,df = rd.read_data(path)
    patients,lesions,classes, classes_ratio = org.organize_data(grouped_data)

    labels = [0,1,2]
    for patient in classes:
        for label in labels:
            if label not in classes[patient]:
                classes[patient][label] = 0

    total_lesions = calculate_total_lesions(classes)
    print("total :",total_lesions)
    target_train , target_test = determine_target_counts(total_lesions)
    print("target train :",target_train)
    print("target test :",target_test)

    max_iterations = 50000

    start = time.time()
    train_patients, test_patients = compute_PL_new(patients, labels, classes,max_iterations,target_train,target_test)
    end = time.time()
    print("Execution time :",end-start)
    return train_patients, test_patients
