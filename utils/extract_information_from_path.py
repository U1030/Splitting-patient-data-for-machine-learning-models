import re

def extract_patient_from_path(path,patients):
    patient = [id for id in patients if id in path ]
    if len(patient) != 0:
        return patient[0]
    else :
        return None

def extract_patient_id_from_path_when_folder_name_different_from_texture_session(path,id_patient_position_in_path):
    path = path.replace("\\","/")   
    id = path.split(sep="/")[id_patient_position_in_path]
    return id

def extract_mask_name(file):
    mask_splited = file.split(sep='_')[:6]
    mask_name = '_'.join(mask_splited) 
    return mask_name 

def extract_date_from_path(string):
    pattern = r'E\d+' 
    match = re.search(pattern, string)
    if match :
        return match.group(0) 
    else : 
        return None