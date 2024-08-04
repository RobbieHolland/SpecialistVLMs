import time 
import pandas as pd

def merge_unique(left_df, right_df, on):
    if left_df[on].isin(right_df[on]).any():
        return left_df.merge(right_df.drop(columns=right_df.columns.intersection(left_df.columns).difference([on])), on=on, how='left', suffixes=('', ''))
    return left_df

def try_load(file_path):
    while True:
        try:
            new_dataframe = pd.read_pickle(file_path)
            break  # If successful, break out of the loop
        except Exception as e:  # Catch any exception. Be careful, this will catch any type of exception!
            print(f"Error reading file: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    return new_dataframe
