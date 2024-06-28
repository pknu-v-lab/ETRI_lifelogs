from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from args_config import get_args

import pandas as pd
import glob
import os

args = get_args()

SELECTED_FEATURES = {
    'sum_values': None,
    'median': None,
    'mean': None,
    'length': None,
    'standard_deviation': None,
    'variance': None,
    'root_mean_square': None,
    'maximum': None,
    'absolute_maximum': None,
    'minimum': None }


def get_user_ids(file_path):
    temp = pd.read_parquet(file_path)
    ids = temp['subject_id'].unique()

    return ids


def get_hr_user_features(file_path, user_id, ts_path):
    hr_temp_df = pd.read_parquet(file_path)

    user_hr_data = hr_temp_df[hr_temp_df['subject_id'] == user_id]
    user_hr_data['timestamp'] = pd.to_datetime(user_hr_data['timestamp'])
    user_hr_data['datetime'] = user_hr_data['timestamp'].dt.floor('10T')
    user_hr_data['min'] = user_hr_data['timestamp'].dt.minute // 10
    user_hr_data = user_hr_data.rename(columns={'heart_rate': 'avg_hr'})

    extracted_hr_features = extract_features(user_hr_data,
                                                column_id='datetime',
                                                column_sort='min',
                                                column_value='avg_hr',
                                                default_fc_parameters=SELECTED_FEATURES)

    imputed_hr_features = impute(extracted_hr_features)
    imputed_hr_features = imputed_hr_features.reset_index().rename(columns={'index': 'datetime'})

    if not os.path.exists(ts_path):
        os.makedirs(ts_path)
        
    imputed_hr_features.to_csv(f'{ts_path}/hr_{user_id}.csv', index=False)

    return ts_path


def get_acc_user_features(file_path, user_id, ts_path):
    acc_temp_data = pd.read_parquet(file_path)

    user_acc_data = acc_temp_data[acc_temp_data['subject_id'] == user_id]
    user_acc_data['timestamp'] = pd.to_datetime(user_acc_data['timestamp'])
    user_acc_data.set_index('timestamp', inplace=True)
    user_acc_avg_data = user_acc_data.resample('T').mean().dropna().reset_index()
    user_acc_avg_data['datetime'] = user_acc_avg_data['timestamp'].dt.floor('10T')
    user_acc_avg_data['min'] = user_acc_avg_data['timestamp'].dt.minute // 10
    user_acc_avg_data = user_acc_avg_data.rename(columns={'x': 'avg_x', 'y': 'avg_y', 'z': 'avg_z'})

    user_acc_avg_data = user_acc_avg_data.melt(id_vars=['timestamp', 'datetime', 'min'],
                                            value_vars=['avg_x', 'avg_y', 'avg_z'],
                                            var_name='variable',
                                            value_name='value')

    extracted_acc_features = extract_features(user_acc_avg_data,
                                                column_id='datetime',
                                                column_sort='min',
                                                column_kind='variable',
                                                column_value='value',
                                                default_fc_parameters=SELECTED_FEATURES)
    
    imputed_acc_features = impute(extracted_acc_features)
    imputed_acc_features = imputed_acc_features.reset_index().rename(columns={'index': 'datetime'})
    
    if not os.path.exists(ts_path):
        os.makedirs(ts_path)

    imputed_acc_features.to_csv(f'{ts_path}/acc_{user_id}.csv', index=False)

    return ts_path


def get_gps_user_features(file_path, user_id, ts_path):
    gps_temp_data = pd.read_parquet(file_path)
    user_gps_data = gps_temp_data[gps_temp_data['subject_id'] == user_id]

    user_gps_data = user_gps_data.drop(columns=['speed'])
    user_gps_data = user_gps_data.drop(columns=['altitude'])

    user_gps_data['timestamp'] = pd.to_datetime(user_gps_data['timestamp'])
    user_gps_data.set_index('timestamp', inplace=True)
    user_gps_avg_data = user_gps_data.resample('T').mean().dropna().reset_index()
    user_gps_avg_data['datetime'] = user_gps_avg_data['timestamp'].dt.floor('10T')
    user_gps_avg_data['min'] = user_gps_avg_data['timestamp'].dt.minute // 10

    user_gps_avg_data = user_gps_avg_data.rename(columns={'latitude': 'avg_lat', 'longitude': 'avg_lon'})

    user_gps_avg_data = user_gps_avg_data.melt(id_vars=['timestamp', 'datetime', 'min'],
                                                value_vars=['avg_lat', 'avg_lon'],
                                                var_name='variable',
                                                value_name='value')

    extracted_gps_features = extract_features(user_gps_avg_data,
                                                column_id='datetime',
                                                column_sort='min',
                                                column_kind='variable',
                                                column_value='value',
                                                default_fc_parameters=SELECTED_FEATURES)

    imputed_gps_features = impute(extracted_gps_features)
    imputed_gps_features = imputed_gps_features.reset_index().rename(columns={'index': 'datetime'})
    
    if not os.path.exists(ts_path):
        os.makedirs(ts_path)
        
    imputed_gps_features.to_csv(f'{ts_path}/gps_{user_id}.csv', index=False)

    return ts_path


def get_ts_features(root_path):
    root_path = root_path+'/*'
    data_paths = glob.glob(root_path)
    
    if "val" in root_path:
        ts_path = args.train_ts_data_root
    else:
        ts_path = args.test_ts_data_root
        
    acc_data_path = []

    for data_path in data_paths:        
        if "heart_rate" in data_path:
            ids = get_user_ids(data_path)
            hr_data_path = data_path
        elif "m_acc" in data_path:
            acc_data_path.append(data_path)
        elif "m_gps" in data_path:
            gps_data_path = data_path
    
    for user_id in ids:
        get_hr_user_features(hr_data_path, user_id, ts_path)
        get_gps_user_features(gps_data_path, user_id, ts_path)
        
        for p in acc_data_path:
            if f"part_{user_id}" in p:
                get_acc_user_features(p, user_id, ts_path)
        
        merged_path = ts_merge(user_id, ts_path)

    return merged_path        


def ts_merge(user_id, ts_path):

    merged_path = ts_path + 'merged'
    
    if not os.path.exists(merged_path):
        os.makedirs(merged_path)
    
    hr = pd.read_csv(f'{ts_path}/hr_{user_id}.csv')
    hr.set_index('datetime', inplace=True)
    
    acc = pd.read_csv(f'{ts_path}/acc_{user_id}.csv')
    acc.set_index('datetime', inplace=True)
    
    gps = pd.read_csv(f'{ts_path}/gps_{user_id}.csv')
    gps.set_index('datetime', inplace=True)
    
    merged_df = hr.merge(acc, left_index=True, right_index=True, suffixes=('', '_acc'), how='outer').merge(gps, left_index=True, right_index=True, suffixes=('', '_gps'), how='outer')
    merged_df.reset_index(inplace=True)
    merged_df = merged_df.fillna(0)
    merged_df.to_csv(f"{merged_path}/merged_{user_id}.csv", index=False)
    # print(f"Saved {len(merged_df)} {user_id} csv file.")
    
    return merged_path

