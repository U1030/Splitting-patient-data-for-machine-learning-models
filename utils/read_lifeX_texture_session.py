import pandas as pd

def read_lifeX_feature_extraction(path_TS_lifeX):
    ts_data_lifeX = pd.read_csv(path_TS_lifeX, skiprows=2, encoding='utf-8',encoding_errors='replace')
    return ts_data_lifeX