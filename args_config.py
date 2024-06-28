import argparse
from model import *


def get_args():
    parser = argparse.ArgumentParser(description='ETRI LIFELOG2024')

    # 입력받을 요소
    parser.add_argument('--train_data_root', type=str, default='./human2024/val_dataset/')
    parser.add_argument('--label_path', type=str, default='./human2024/val_label.csv')
    parser.add_argument('--test_data_root', type=str, default='./human2024/test_dataset/')
    parser.add_argument('-w', '--weight', type=str, help='weight path', default='./weights/combined_model.pth')

    # model num, threshold 바꾸기
    parser.add_argument('--model_num', type=int, default=9, help='ensemble model number')
    parser.add_argument('--name', type=str, default='Test' ,help='path name')
    parser.add_argument('--root', type=str, default='./weights/', help='weight save path')
    parser.add_argument('--train_ts_data_root', type=str, default='./feature_data/train_ts/')
    parser.add_argument('--test_ts_data_root', type=str, default='./feature_data/test_ts/')

    # Path
    parser.add_argument('--save_path', type=str, default='./figures/')
    parser.add_argument('--output_save_root', type=str, default='./')

    # model parameters
    parser.add_argument('--lr' , default=0.01, help='learning rate')
    parser.add_argument('--epochs', default=80, help='epoch')
    parser.add_argument('--gamma', default=0.1, help='gamma value of learning scheduler')
    parser.add_argument('--model', default=LSTM, help='option : LSTM, BILSTM , GRU , BIGRU')
    parser.add_argument('--scheduler', default=False, help='True : Cosin Annealing | False : Multi Step')
    parser.add_argument('--step', type=list, default=[20], help='step epochs')
    parser.add_argument('--threshold', default=0.59 ,help='threshold')
    
    # Augmentation
    parser.add_argument('--transforms', default=True, help='Data augmentation')
    parser.add_argument('--time_shift', default=6, help='time shift level')
    parser.add_argument('--noise_level', default=0.003, help='noise level')
    parser.add_argument('--idx_percent', default=0.2, help='Sequence Percent')

    # Base
    parser.add_argument('--inputs_size', default=60 ,help='inputs size')
    parser.add_argument('--output_size', default=7 ,help='Output size')
    parser.add_argument('--decay', default=5e-5, help='weight decay')
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--duplicate', default=True, help='Class Augmentation')
    parser.add_argument('--hidden', default=32 ,help='hidden size')
    parser.add_argument('--num_layers', default=2, help='num layers')
    
    parser.add_argument('--knum', default = 3, help = 'K-fold Num')
    
    args = parser.parse_args()
    
    return args
