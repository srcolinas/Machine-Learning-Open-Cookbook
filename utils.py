import pandas as pd

def load_toy_datasets(filepath, feature_names, label_name='y'):
    """Loads a toy dataset from its filepath.

    Args:
        filepath (str): path to the toy dataset
        feature_names (list or None): column names to use as features.
        label_name (str): name of the output variable, defaults to 'y'.
    
    Returns:
    (tuple): Numpy arrays corresponding to X_train, y_train, X_test, y_test

    """
    df = pd.read_csv(filepath)
    df_train = df[df['split'] == 'train']
    X_train, y_train = df_train[feature_names].values, df_train[label_name].values
    df_test = df[df['split'] == 'test']
    X_test, y_test = df_test[feature_names].values, df_test[label_name].values
    return X_train, y_train, X_test, y_test