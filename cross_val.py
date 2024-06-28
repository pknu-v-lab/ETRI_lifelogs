from train import *
from transforms import *
from args_config import get_args
from submit_test import *
from multiprocessing import freeze_support



if __name__ == '__main__':
    freeze_support()
    args = get_args()
    cross_val(args)