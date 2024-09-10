import os 
import pandas as pd

def explore_folders(folder_path, folder_paths): 
    if any(file.endswith(".nii.gz") for file in os.listdir(folder_path)): 
        folder_paths.append(folder_path)             
        return
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            explore_folders(subfolder_path, folder_paths)


def extract_folders_path(data_path):
    folder_paths = []       
    for patient in os.listdir(data_path):
        patient_path = os.path.join(data_path, patient)        
        if os.path.isdir(patient_path):                       
            explore_folders(patient_path, folder_paths)
    return folder_paths

def extract_patient_from_path(path,patients):
    patient = [id for id in patients if id in path ]
    return patient[0]


def extract_patient_id_from_path_when_folder_name_different_from_ts(path,id_patient_position_in_path):
    path = path.replace("\\","/")   
    id = path.split(sep="/")[id_patient_position_in_path]
    return id

def extract_mask_name(file):
    mask_splited = file.split(sep='_')[:6]
    mask_name = '_'.join(mask_splited) 
    return mask_name 

def find_all_nifti_paths(data_path,id_patient_position_in_path):
    print("=> Extracting folder path")
    folder_paths = extract_folders_path(data_path)
    data_rows = []    
    for path in folder_paths:        
        patient = extract_patient_id_from_path_when_folder_name_different_from_ts(path, id_patient_position_in_path)        
        masks = []
        image_path = None        
        for file in os.listdir(path):
            if file.endswith("nii.gz"):
                if "mask" in file:
                    mask_name = extract_mask_name(file)              
                    masks.append((mask_name, os.path.join(path, file)))                   
                else:
                    image_path = os.path.join(path, file)        
        for mask_name, mask_path in masks:
            data_rows.append({
                "patient_id": patient,
                "roiname": mask_name,
                "image_path": image_path,
                "mask_path": mask_path
            })    
    return pd.DataFrame(data_rows)

