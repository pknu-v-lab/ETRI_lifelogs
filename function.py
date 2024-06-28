import pandas as pd
import glob
import os
import re
from sklearn.preprocessing import MinMaxScaler
import torch
import logging
import torch.nn as nn
import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

scaler = MinMaxScaler()

def get_data(root):
    file_paths = glob.glob(root+'/*')
    file_paths.sort()
    data_list = []

    for file_path in file_paths:
        find_id = re.search(r'merged_(\d+)\.csv', file_path)
        if find_id:
            user_id = int(find_id.group(1))
            
        tmp_df = pd.read_csv(file_path)

        for _, row in tmp_df.iterrows():
            data = {
                'user': user_id,
                'timestamp': row['datetime'],
                'abs_max': row['avg_hr__absolute_maximum'],
                'length': row['avg_hr__length'],
                'hr_max': row['avg_hr__maximum'],
                'hr_mean': row['avg_hr__mean'],
                'median': row['avg_hr__median'],
                'min': row['avg_hr__minimum'],
                'rme': row['avg_hr__root_mean_square'],
                'sd': row['avg_hr__standard_deviation'],
                'sum_value': row['avg_hr__sum_values'],
                'var': row['avg_hr__variance'],
                'x_sumval': row['avg_x__sum_values'],
                'x_median': row['avg_x__median'],
                'x_mean': row['avg_x__mean'],
                'x_length': row['avg_x__length'],
                'x_sd': row['avg_x__standard_deviation'],
                'x_var': row['avg_x__variance'],
                'x_rme': row['avg_x__root_mean_square'],
                'x_maximum': row['avg_x__maximum'],
                'x_abs_max': row['avg_x__absolute_maximum'],
                'x_min': row['avg_x__minimum'],
                'y_sumval': row['avg_y__sum_values'],
                'y_median': row['avg_y__median'],
                'y_mean': row['avg_y__mean'],
                'y_length': row['avg_y__length'],
                'y_sd': row['avg_y__standard_deviation'],
                'y_var': row['avg_y__variance'],
                'y_rme': row['avg_y__root_mean_square'],
                'y_maximum': row['avg_y__maximum'],
                'y_abs_max': row['avg_y__absolute_maximum'],
                'y_minimum': row['avg_y__minimum'],
                'z_sumval': row['avg_z__sum_values'],
                'z_median': row['avg_z__median'],
                'z_mean': row['avg_z__mean'],
                'z_length': row['avg_z__length'],
                'z_sd': row['avg_z__standard_deviation'],
                'z_var': row['avg_z__variance'],
                'z_rme': row['avg_z__root_mean_square'],
                'z_maximum': row['avg_z__maximum'],
                'z_abs_max': row['avg_z__absolute_maximum'],
                'z_minimum': row['avg_z__minimum'],
                'lat_sumval': row['avg_lat__sum_values'],
                'lat_median': row['avg_lat__median'],
                'lat_mean': row['avg_lat__mean'],
                'lat_length': row['avg_lat__length'],
                'lat_sd': row['avg_lat__standard_deviation'],
                'lat_var': row['avg_lat__variance'],
                'lat_rme': row['avg_lat__root_mean_square'],
                'lat_maximum': row['avg_lat__maximum'],
                'lat_abs_max': row['avg_lat__absolute_maximum'],
                'lat_minimum': row['avg_lat__minimum'],
                'lon_sumval': row['avg_lon__sum_values'],
                'lon_median': row['avg_lon__median'],
                'lon_mean': row['avg_lon__mean'],
                'lon_length': row['avg_lon__length'],
                'lon_sd': row['avg_lon__standard_deviation'],
                'lon_var': row['avg_lon__variance'],
                'lon_rme': row['avg_lon__root_mean_square'],
                'lon_maximum': row['avg_lon__maximum'],
                'lon_abs_max': row['avg_lon__absolute_maximum'],
                'lon_minimum': row['avg_lon__minimum']
            }

            data_list.append(data)
    
    df = pd.DataFrame(data_list)

    variables_to_scale = [
        'abs_max', 'length', 'hr_mean', 'median', 'min', 'rme', 'sd', 'sum_value', 'var', 'hr_max',
        'x_sumval', 'x_median', 'x_mean', 'x_length', 'x_sd', 'x_var', 'x_rme', 'x_maximum', 'x_abs_max', 'x_min',
        'y_sumval', 'y_median', 'y_mean', 'y_length', 'y_sd', 'y_var', 'y_rme', 'y_maximum', 'y_abs_max', 'y_minimum',
        'z_sumval', 'z_median', 'z_mean', 'z_length', 'z_sd', 'z_var', 'z_rme', 'z_maximum', 'z_abs_max', 'z_minimum',
        'lat_sumval', 'lat_median', 'lat_mean', 'lat_length', 'lat_sd', 'lat_var', 'lat_rme', 'lat_maximum', 'lat_abs_max', 'lat_minimum',
        'lon_sumval', 'lon_median', 'lon_mean', 'lon_length', 'lon_sd', 'lon_var', 'lon_rme', 'lon_maximum', 'lon_abs_max', 'lon_minimum'
    ]

    for var in variables_to_scale:
        df[var] = scaler.fit_transform(df[[var]])
    
    return df

def merge_data(temp_data, label_data):
    temp_data['timestamp'] = pd.to_datetime(temp_data['timestamp'])
    temp_data['date'] = temp_data['timestamp'].dt.normalize()

    temp_sequences = temp_data.groupby(['user', 'date']).agg({
        'abs_max': list,
        'length': list,
        'hr_max': list,
        'hr_mean': list,
        'median': list,
        'min': list,
        'rme': list,
        'sd': list,
        'sum_value': list,
        'var': list,
        'x_sumval': list,
        'x_median': list,
        'x_mean': list,
        'x_length': list,
        'x_sd': list,
        'x_var': list,
        'x_rme': list,
        'x_maximum': list,
        'x_abs_max': list,
        'x_min': list,
        'y_sumval': list,
        'y_median': list,
        'y_mean': list,
        'y_length': list,
        'y_sd': list,
        'y_var': list,
        'y_rme': list,
        'y_maximum': list,
        'y_abs_max': list,
        'y_minimum': list,
        'z_sumval': list,
        'z_median': list,
        'z_mean': list,
        'z_length': list,
        'z_sd': list,
        'z_var': list,
        'z_rme': list,
        'z_maximum': list,
        'z_abs_max': list,
        'z_minimum': list,
        'lat_sumval': list,
        'lat_median': list,
        'lat_mean': list,
        'lat_length': list,
        'lat_sd': list,
        'lat_var': list,
        'lat_rme': list,
        'lat_maximum': list,
        'lat_abs_max': list,
        'lat_minimum': list,
        'lon_sumval': list,
        'lon_median': list,
        'lon_mean': list,
        'lon_length': list,
        'lon_sd': list,
        'lon_var': list,
        'lon_rme': list,
        'lon_maximum': list,
        'lon_abs_max': list,
        'lon_minimum': list
    }).reset_index()

    label_data.reset_index(inplace=True)
    label_data['date'] = pd.to_datetime(label_data['timestamp']).dt.normalize()
    label_data['user'] = label_data['subject_id']
    label_data.set_index(['user', 'date'], inplace=True)

    merged_data = temp_sequences.set_index(['user', 'date']).join(label_data[['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']], how='inner')

    X = merged_data[[
        'abs_max', 'length', 'hr_max', 'hr_mean', 'median', 'min', 'rme', 'sd', 'sum_value', 'var', 
        'x_sumval', 'x_median', 'x_mean', 'x_length', 'x_sd', 'x_var', 'x_rme', 'x_maximum', 'x_abs_max', 'x_min',
        'y_sumval', 'y_median', 'y_mean', 'y_length', 'y_sd', 'y_var', 'y_rme', 'y_maximum', 'y_abs_max', 'y_minimum',
        'z_sumval', 'z_median', 'z_mean', 'z_length', 'z_sd', 'z_var', 'z_rme', 'z_maximum', 'z_abs_max', 'z_minimum',
        'lat_sumval', 'lat_median', 'lat_mean', 'lat_length', 'lat_sd', 'lat_var', 'lat_rme', 'lat_maximum', 'lat_abs_max', 'lat_minimum',
        'lon_sumval', 'lon_median', 'lon_mean', 'lon_length', 'lon_sd', 'lon_var', 'lon_rme', 'lon_maximum', 'lon_abs_max', 'lon_minimum'
    ]].values.tolist()
    
    y = merged_data[['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']].values.tolist()

    return X, y

def load_label_data(filepath):
    label_data = pd.read_csv(filepath)
    label_data['timestamp'] = pd.to_datetime(label_data['date'])
    label_data.set_index(['subject_id', 'timestamp'], inplace=True)
    
    return label_data
    
def get_test(root):
    file_paths = glob.glob(root+'/*')
    file_paths.sort()
    data_list = []
    lst = []
    
    for file_path in file_paths:
        find_id = re.search(r'merged_(\d+)\.csv', file_path)
        if find_id:
            user_id = int(find_id.group(1))
        
        tmp_df = pd.read_csv(file_path)

        for _, row in tmp_df.iterrows():
            data = {
                'user': user_id,
                'timestamp': row['datetime'],
                'abs_max': row['avg_hr__absolute_maximum'],
                'length': row['avg_hr__length'],
                'hr_max': row['avg_hr__maximum'],
                'hr_mean': row['avg_hr__mean'],
                'median': row['avg_hr__median'],
                'min': row['avg_hr__minimum'],
                'rme': row['avg_hr__root_mean_square'],
                'sd': row['avg_hr__standard_deviation'],
                'sum_value': row['avg_hr__sum_values'],
                'var': row['avg_hr__variance'],
                'x_sumval': row['avg_x__sum_values'],
                'x_median': row['avg_x__median'],
                'x_mean': row['avg_x__mean'],
                'x_length': row['avg_x__length'],
                'x_sd': row['avg_x__standard_deviation'],
                'x_var': row['avg_x__variance'],
                'x_rme': row['avg_x__root_mean_square'],
                'x_maximum': row['avg_x__maximum'],
                'x_abs_max': row['avg_x__absolute_maximum'],
                'x_min': row['avg_x__minimum'],
                'y_sumval': row['avg_y__sum_values'],
                'y_median': row['avg_y__median'],
                'y_mean': row['avg_y__mean'],
                'y_length': row['avg_y__length'],
                'y_sd': row['avg_y__standard_deviation'],
                'y_var': row['avg_y__variance'],
                'y_rme': row['avg_y__root_mean_square'],
                'y_maximum': row['avg_y__maximum'],
                'y_abs_max': row['avg_y__absolute_maximum'],
                'y_minimum': row['avg_y__minimum'],
                'z_sumval': row['avg_z__sum_values'],
                'z_median': row['avg_z__median'],
                'z_mean': row['avg_z__mean'],
                'z_length': row['avg_z__length'],
                'z_sd': row['avg_z__standard_deviation'],
                'z_var': row['avg_z__variance'],
                'z_rme': row['avg_z__root_mean_square'],
                'z_maximum': row['avg_z__maximum'],
                'z_abs_max': row['avg_z__absolute_maximum'],
                'z_minimum': row['avg_z__minimum'],
                'lat_sumval': row['avg_lat__sum_values'],
                'lat_median': row['avg_lat__median'],
                'lat_mean': row['avg_lat__mean'],
                'lat_length': row['avg_lat__length'],
                'lat_sd': row['avg_lat__standard_deviation'],
                'lat_var': row['avg_lat__variance'],
                'lat_rme': row['avg_lat__root_mean_square'],
                'lat_maximum': row['avg_lat__maximum'],
                'lat_abs_max': row['avg_lat__absolute_maximum'],
                'lat_minimum': row['avg_lat__minimum'],
                'lon_sumval': row['avg_lon__sum_values'],
                'lon_median': row['avg_lon__median'],
                'lon_mean': row['avg_lon__mean'],
                'lon_length': row['avg_lon__length'],
                'lon_sd': row['avg_lon__standard_deviation'],
                'lon_var': row['avg_lon__variance'],
                'lon_rme': row['avg_lon__root_mean_square'],
                'lon_maximum': row['avg_lon__maximum'],
                'lon_abs_max': row['avg_lon__absolute_maximum'],
                'lon_minimum': row['avg_lon__minimum']
            }

            data_list.append(data)
    
    df = pd.DataFrame(data_list)

    variables_to_scale = [
        'abs_max', 'length', 'hr_mean', 'median', 'min', 'rme', 'sd', 'sum_value', 'var', 'hr_max',
        'x_sumval', 'x_median', 'x_mean', 'x_length', 'x_sd', 'x_var', 'x_rme', 'x_maximum', 'x_abs_max', 'x_min',
        'y_sumval', 'y_median', 'y_mean', 'y_length', 'y_sd', 'y_var', 'y_rme', 'y_maximum', 'y_abs_max', 'y_minimum',
        'z_sumval', 'z_median', 'z_mean', 'z_length', 'z_sd', 'z_var', 'z_rme', 'z_maximum', 'z_abs_max', 'z_minimum',
        'lat_sumval', 'lat_median', 'lat_mean', 'lat_length', 'lat_sd', 'lat_var', 'lat_rme', 'lat_maximum', 'lat_abs_max', 'lat_minimum',
        'lon_sumval', 'lon_median', 'lon_mean', 'lon_length', 'lon_sd', 'lon_var', 'lon_rme', 'lon_maximum', 'lon_abs_max', 'lon_minimum'
    ]

    for var in variables_to_scale:
        df[var] = scaler.fit_transform(df[[var]])
    return df

def save_model(model, learnig_rate, hidden_size, directory,args):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = os.path.join(directory, f'{learnig_rate}_hd{hidden_size}_{args.name}.pth')
    
    torch.save(model.state_dict(), file_path)
    print(f'Model saved to {file_path}')

def setup_logger(log_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  
    
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
            
    log_filename = os.path.join(log_dir , datetime.now().strftime('train_log_%Y-%m-%d_%H-%M-%S.log'))
    
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
                    

def merge_test(temp_data):
    temp_data['timestamp'] = pd.to_datetime(temp_data['timestamp'])
    temp_data['date'] = temp_data['timestamp'].dt.normalize()

    X = []
    
    temp_sequences = temp_data.groupby(['user', 'date']).agg({
        'abs_max': list,
        'length': list,
        'hr_max': list,
        'hr_mean': list,
        'median': list,
        'min': list,
        'rme': list,
        'sd': list,
        'sum_value': list,
        'var': list,
        'x_sumval': list,
        'x_median': list,
        'x_mean': list,
        'x_length': list,
        'x_sd': list,
        'x_var': list,
        'x_rme': list,
        'x_maximum': list,
        'x_abs_max': list,
        'x_min': list,
        'y_sumval': list,
        'y_median': list,
        'y_mean': list,
        'y_length': list,
        'y_sd': list,
        'y_var': list,
        'y_rme': list,
        'y_maximum': list,
        'y_abs_max': list,
        'y_minimum': list,
        'z_sumval': list,
        'z_median': list,
        'z_mean': list,
        'z_length': list,
        'z_sd': list,
        'z_var': list,
        'z_rme': list,
        'z_maximum': list,
        'z_abs_max': list,
        'z_minimum': list,
        'lat_sumval': list,
        'lat_median': list,
        'lat_mean': list,
        'lat_length': list,
        'lat_sd': list,
        'lat_var': list,
        'lat_rme': list,
        'lat_maximum': list,
        'lat_abs_max': list,
        'lat_minimum': list,
        'lon_sumval': list,
        'lon_median': list,
        'lon_mean': list,
        'lon_length': list,
        'lon_sd': list,
        'lon_var': list,
        'lon_rme': list,
        'lon_maximum': list,
        'lon_abs_max': list,
        'lon_minimum': list
    }).reset_index()
    
    for _, row in temp_sequences.iterrows():
        X.append({
            'subject_id': row['user'],
            'date': pd.Timestamp(row['date']),
            'sequence': [
                row['abs_max'], row['length'], row['hr_max'], row['hr_mean'], row['median'], row['min'], row['rme'], row['sd'], row['sum_value'], row['var'],
                row['x_sumval'], row['x_median'], row['x_mean'], row['x_length'], row['x_sd'], row['x_var'], row['x_rme'], row['x_maximum'], row['x_abs_max'], row['x_min'],
                row['y_sumval'], row['y_median'], row['y_mean'], row['y_length'], row['y_sd'], row['y_var'], row['y_rme'], row['y_maximum'], row['y_abs_max'], row['y_minimum'],
                row['z_sumval'], row['z_median'], row['z_mean'], row['z_length'], row['z_sd'], row['z_var'], row['z_rme'], row['z_maximum'], row['z_abs_max'], row['z_minimum'],
                row['lat_sumval'], row['lat_median'], row['lat_mean'], row['lat_length'], row['lat_sd'], row['lat_var'], row['lat_rme'], row['lat_maximum'], row['lat_abs_max'], row['lat_minimum'],
                row['lon_sumval'], row['lon_median'], row['lon_mean'], row['lon_length'], row['lon_sd'], row['lon_var'], row['lon_rme'], row['lon_maximum'], row['lon_abs_max'], row['lon_minimum']
            ]
        })

    return X

def duplicate_labels(x, y):
    targets = [[1, 0, 0, 1, 0, 0, 1], [1, 1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
    positions_dict = {}
    
    for target in targets:
        positions = int([index for index, value in enumerate(y) if value == target][0])
        positions_dict[tuple(target)] = positions
    
    indices_to_duplicate = list(positions_dict.values())

    for idx in indices_to_duplicate:
        x.extend([x[idx]] * 11)
        y.extend([y[idx]] * 11)
    
    return x, y

def cal_mean_var(loader):
    variables_lst = [
        'hr_abs_max', 'hr_length', 'hr_mean', 'hr_median', 'hr_min', 'hr_rme', 'hr_sd', 'hr_sum_value', 'hr_var', 'hr_max',
        'x_sumval', 'x_median', 'x_mean', 'x_length', 'x_sd', 'x_var', 'x_rme', 'x_maximum', 'x_abs_max', 'x_min',
        'y_sumval', 'y_median', 'y_mean', 'y_length', 'y_sd', 'y_var', 'y_rme', 'y_maximum', 'y_abs_max', 'y_minimum',
        'z_sumval', 'z_median', 'z_mean', 'z_length', 'z_sd', 'z_var', 'z_rme', 'z_maximum', 'z_abs_max', 'z_minimum',
        'lat_sumval', 'lat_median', 'lat_mean', 'lat_length', 'lat_sd', 'lat_var', 'lat_rme', 'lat_maximum', 'lat_abs_max', 'lat_minimum',
        'lon_sumval', 'lon_median', 'lon_mean', 'lon_length', 'lon_sd', 'lon_var', 'lon_rme', 'lon_maximum', 'lon_abs_max', 'lon_minimum'
    ]
    
    lst = [[] for _ in range(60)]
    mean_lstt = [[] for _ in range(60)]
    var_lstt = []
    
    for i, (inputs, labels, length) in enumerate(loader):
        for k in range(60):
            lst[k].extend(inputs[0][k].numpy())
        
    for i in range(60):
        mean_lstt[i].append(np.mean(lst[i]))
        var_lstt.append(np.var(lst[i]))
        
    mean_lst = []
    var_lst = []
    
    for col_data in lst:
        mean_lst.append(np.mean(col_data))
        var_lst.append(np.var(col_data))

    columns = [f'{i}' for i in variables_lst] 
    x = np.arange(len(columns))  
    width = 0.5

    fig, ax = plt.subplots(figsize=(15, 8))
    # rects1 = ax.bar(x - width/2, mean_lst, width, label='Mean')
    rects2 = ax.bar(x + width/2, var_lst, width, label='Variance')

    ax.set_xlabel('Columns')
    ax.set_ylabel('Scores')
    ax.set_title('Variance')
    ax.set_xticks(x)
    ax.set_xticklabels(columns, rotation=90)
    ax.legend()

    fig.tight_layout()
    
    return var_lstt