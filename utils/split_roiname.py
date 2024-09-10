import pandas as pd
import re

def find_location(path):   
    parts = path.split('_')
    part3 = parts[3]
    part2 = parts[2]
    if part2 == "ring" or part2 == "tum":
        return part3
    else :
        return part2

def find_VOInum(path):   
    parts = path.split('_')
    return parts[4]

def find_date(input_string):    
    pattern = r'E\d+'   
    match = re.search(pattern, input_string)    
    if match:
        return match.group()
    else:       
        return None
    
def find_voi_type(string):
    if '_tum_' in string :
        return 'tum'
    elif '_ring_' in string:
        return 'ring'

def extract_metadata(data_cleaned):
    data_cleaned['date'] = data_cleaned["roiname"].apply(lambda x : find_date(x))
    data_cleaned['location'] = data_cleaned["roiname"].apply(lambda x : find_location(x))
    data_cleaned['VOInum'] = data_cleaned["roiname"].apply(lambda x : find_VOInum(x))
    return data_cleaned

def extract_metadata_with_voi_type(data_cleaned):
    data_cleaned['date'] = data_cleaned["roiname"].apply(lambda x : find_date(x))
    data_cleaned['location'] = data_cleaned["roiname"].apply(lambda x : find_location(x))
    data_cleaned['VOInum'] = data_cleaned["roiname"].apply(lambda x : find_VOInum(x))
    data_cleaned['VOItype'] = data_cleaned["roiname"].apply(lambda x : find_voi_type(x))
    return data_cleaned
