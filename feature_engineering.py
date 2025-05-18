import pandas as pd

def extract_date_features(df):
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    df['day_of_week_added'] = df['date_added'].dt.dayofweek
    return df

def count_multi_valued_fields(df):
    df['num_cast'] = df['cast'].apply(lambda x: 0 if x == 'Unknown' else len(x.split(',')))
    df['num_categories'] = df['listed_in'].apply(lambda x: len(x.split(',')))
    df['num_countries'] = df['country'].apply(lambda x: 0 if x == 'Unknown' else len(x.split(',')))
    return df

def parse_duration(val):
    if 'Season' in val:
        return {'duration_type': 'Season', 'duration_int': int(val.split()[0])}
    elif 'min' in val:
        return {'duration_type': 'Minutes', 'duration_int': int(val.split()[0])}
    else:
        return {'duration_type': 'Unknown', 'duration_int': 0}

def transform_duration(df):
    parsed = df['duration'].apply(parse_duration)
    df['duration_type'] = parsed.apply(lambda x: x['duration_type'])
    df['duration_int'] = parsed.apply(lambda x: x['duration_int'])
    df.drop(columns='duration', inplace=True)
    return df

def encode_categorical(df):
    df_encoded = pd.get_dummies(df, columns=['type', 'rating', 'duration_type'], drop_first=True)
    return df_encoded
