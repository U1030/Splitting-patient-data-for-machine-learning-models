import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import os
import time
import explore_folders_to_find_a_type_of_file as find_paths
import extract_information_from_path as get_from_path
import parallele_processing as par


def find_all_nifti_paths(data_path,id_patient_position_in_path,patients_df,patients_id_same=True):
    folder_paths = find_paths.extract_folders_path(data_path, type="nifti")
    data_rows = []
    for path in folder_paths:
        print("folder path :", path)
        if patients_id_same:
            patient = get_from_path.extract_patient_from_path(path, patients_df)
            if patient == None:
                print("WARNING not in patients ! ")
                patient = get_from_path.extract_patient_id_from_path_when_folder_name_different_from_texture_session(path,id_patient_position_in_path)               
        else :
            patient = get_from_path.extract_patient_id_from_path_when_folder_name_different_from_texture_session(path,id_patient_position_in_path)
        print("patient :", patient)
        date = get_from_path.extract_date_from_path(path)
        if date:
            print("date :", date)
        
        masks = []
        image_path = None
        
        for file in os.listdir(path):
            if file.endswith("nii.gz"):
                if "mask" in file:
                    mask_splited = file.split(sep='_')[:5]
                    mask_name = '_'.join(mask_splited)
                    masks.append((mask_name, os.path.join(path, file)))
                    print("roiname :", mask_name)
                else:
                    image_path = os.path.join(path, file)
                    print("image :", file)
        
        for mask_name, mask_path in masks:
            data_rows.append({
                "patient_id": patient,
                "date": date,
                "roiname": mask_name,
                "image_path": image_path,
                "mask_path": mask_path,                
            })
    
    return pd.DataFrame(data_rows)

def extract_max_diameter_df(row):
    _, row_data = row 
    path_im = row_data['image_path']
    path_mask = row_data['mask_path']  
    if path_im != None and path_mask != None :  
        image = sitk.ReadImage(path_im)
        mask = sitk.ReadImage(path_mask)
        extractor = featureextractor.RadiomicsFeatureExtractor()
        features = extractor.execute(image, mask)
        row_diam = 'original_shape_Maximum2DDiameterSlice'
        max_diameter = features[row_diam]
        return max_diameter
    else :
        return "None"


def find_image_nifti(path):
    files = os.listdir(path)
    filename = None
    for file in files:
        if file.endswith(".nii.gz") and "mask" not in file:
            filename = file
            break  # Exit the loop once a suitable file is found
    return filename

def match_paths(df,folder_path,pos_id,patients_id_same):
    df_with_path_nifti = find_all_nifti_paths(folder_path,pos_id,df["PatientID"],patients_id_same)
    print(df_with_path_nifti[df_with_path_nifti.isna().any(axis=1)])
    df_with_path_nifti.loc[df_with_path_nifti['date'].isna(), 'date'] = df_with_path_nifti.loc[df_with_path_nifti['date'].isna(), 'roiname'].apply(get_from_path.extract_date_from_path)
    print("rows with None values :")
    print(df_with_path_nifti[df_with_path_nifti.isna().any(axis=1)])
    df['ROIname'] = df['ROIname'].str.replace("'", "")
    df["roiname"] = df["ROIname"].apply(lambda x: get_from_path.extract_mask_name(x))
    df_with_path_nifti.rename(columns={'patient_id': 'PatientID', 'date': 'Date'}, inplace=True)
    merged_df = pd.merge(df, df_with_path_nifti, on=['PatientID', 'Date', 'roiname'], how='inner')
    #print(merged_df)
    print(len(df),len(merged_df))
    return merged_df

def main(df,folder_path,pos_id,patients_id_same):
    merged_data = match_paths(df,folder_path,pos_id,patients_id_same)
    print(merged_data)
    start = time.time()
    results = par .parallelize_maintain_index(merged_data, extract_max_diameter_df)
    end = time.time()
    print("Time to extract diameter feature: {:.2f} seconds".format(end - start))
    results_df = pd.DataFrame(results, columns=['Maximum2DDiameterSlice'])  
    df_with_max_diameter = pd.concat([merged_data, results_df], axis=1)
    df_with_max_diameter.to_csv('CD8_scores_and_diameters.csv', index=False)
    return df_with_max_diameter


def insert_space(string):
  new_string = string[:4] + "_" + string[4:9] + "_" + string[9:]
  return new_string

# special case ACSES patient folder is different from patient id
#df["PatientID"] = df["PatientID"].apply(lambda x : insert_space(x))

