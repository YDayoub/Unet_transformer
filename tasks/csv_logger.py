import pandas as pd
import os
def log_data(file_path: str, data: dict):
    def preprocess_data(data):
        d = {}
        for key,val in data.items():
            d[key] = str(val)
        return d
    data_processed = preprocess_data(data)
    if not os.path.isfile(file_path):
        df = pd.DataFrame(data_processed,index=[0])
        df.to_csv(file_path, index=False)
        return
    df = pd.read_csv(file_path)
    df2 = pd.DataFrame(data_processed, index=[0])
    df = pd.concat([df,df2],axis=0)
    df.to_csv(file_path,  index=False)