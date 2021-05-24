import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

fn = 'ibm_hr/{}.pkl'

RAW_VALUES = {
    'Education': {
        1: 'Below College',
        2: 'College',
        3: 'Bachelor',
        4: 'Master',
        5: 'Doctor'
    },
    'EnvironmentSatisfaction': {
        1: 'Low',
        2: 'Medium',
        3: 'High',
        4: 'Very High'
    },
    'JobInvolvement': {
        1: 'Low',
        2: 'Medium',
        3: 'High',
        4: 'Very High'
    },
    'JobSatisfaction': {
        1: 'Low',
        2: 'Medium',
        3: 'High',
        4: 'Very High'
    },
    'PerformanceRating': {
        1: 'Low',
        2: 'Good',
        3: 'Excellent',
        4: 'Outstanding'
    },
    'RelationshipSatisfaction': {
        1: 'Low',
        2: 'Medium',
        3: 'High',
        4: 'Very High'
    },
    'WorkLifeBalance': {
        1: 'Bad',
        2: 'Good',
        3: 'Better',
        4: 'Best'
    },
}


def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def write_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':
    df = pd.read_csv('ibm_hr.csv')
    df = df.drop(columns=[
        'StandardHours',
        'EmployeeCount',
        'Over18',
        'EmployeeNumber'
    ])
    # df['PerformanceRating'] = df['PerformanceRating'].astype(str)
    for c in RAW_VALUES:
        df[c] = df[c].apply(lambda x: RAW_VALUES[c][x])
    encoder = LabelEncoder()
    df['Attrition'] = encoder.fit_transform(df['Attrition'])
    categorical_features = []
    numerical_features = []
    for column in df.columns:
        if df[column].dtype == object:
            categorical_features.append(column)
        else:
            numerical_features.append(column)
    numerical_features.remove('Attrition')
    X_raw = df.drop(columns=['Attrition'])
    y_raw = df['Attrition']
    std_ct = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
    X_std = std_ct.fit_transform(X_raw)
    norm_ct = ColumnTransformer([
        ('num', Normalizer(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
    X_norm = norm_ct.fit_transform(X_raw)
    # Save data
    write_pickle(X_raw, fn.format('X_df'))
    write_pickle(y_raw, fn.format('y_raw'))
    write_pickle(X_std, fn.format('X_std'))
    write_pickle(X_norm, fn.format('X_norm'))
    write_pickle(std_ct, fn.format('std_ct'))
    write_pickle(norm_ct, fn.format('norm_ct'))
