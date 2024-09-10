def organize_data(grouped_data):
    patients = []
    lesions = {}
    classes = {}
    for patient, data in grouped_data:  
        patient = str(patient[0])
        patients.append(patient)     
        nb_lesions = len(data)
        lesions[patient] = nb_lesions    
        class_count = data["label"].value_counts()
        class_dict = class_count.to_dict()   
        classes[patient] = class_dict
    classes_ratio = {}
    for patient, classes_dict in classes.items():
        total_eff = sum(classes_dict.values())  
        classes_ratio[patient] = {class_id: eff / total_eff for class_id, eff in classes_dict.items()}
    return patients,lesions,classes,classes_ratio