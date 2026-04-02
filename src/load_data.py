import os
import pandas as pd
# We have to read the data and split the id from the DNA sequence
def get_data_path(filename):

    "Obtains the absolute row from data files."

    base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, '..', 'data', filename)



def load_viral_dataset(filename):

    path = get_data_path(filename)

    print(f"Reading {filename}...")
    print(f"DEBUG: Buscando el archivo en: {os.path.abspath(path)}")

    

    # paper says they don't have header, the label is in the last row (header=None)

    df = pd.read_csv(path, header=None)

    

    X = df.iloc[:, :-1].values # Sequences

    y = df.iloc[:, :-1].values  # Lables (Last row)

    

    return X, y



if __name__ == "__main__":

    # Quick test to verify it's working

    try:

        X, y = load_viral_dataset('fullset_train.csv')

        print(f"Success. X: {X.shape}, y: {y.shape}")

    except Exception as e:

        print(f"Error: No files found on /data \n{e}")