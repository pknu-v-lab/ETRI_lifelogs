import torch
import numpy as np 
import random
from args_config import get_args

args = get_args()

# 분산 값 리스트
var_lst = [0.07052669, 0.13565351, 0.07052669, 0.064132504, 0.06522641, 0.052200224, 0.0652225, 0.011365964, 0.065793954,
           0.0034189492, 0.017007127, 0.016689967, 0.015759313, 0.058754336, 0.017783117, 0.00411649, 0.01895773, 0.019887708,
           0.02761522, 0.016660834, 0.040470764, 0.04319445, 0.040777974, 0.058754336, 0.029224386, 0.008902466, 0.05294349,
           0.053811498, 0.042857844, 0.037990175, 0.12300801, 0.13088384, 0.12381314, 0.058754336, 0.032495745, 0.013085667, 0.102444224,
           0.12409156, 0.08076377, 0.13312536, 0.10298103, 0.12956196, 0.12956066, 0.23738001, 0.00046805368, 0.00018106205, 0.12956057, 0.12946095,
           0.12946095, 0.12957974, 0.14748971, 0.16805732, 0.16806282, 0.23738001, 0.00040611337, 0.000114110844, 0.16806115, 0.16809388, 0.16809388, 0.16802664]

def time_shift(x, shift=6):
    rand_p = random.uniform(0, 1)
    if rand_p <= 0.5:
        x = np.roll(x, shift=args.time_shift, axis=0)
        x = torch.from_numpy(x)
    return x

class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def add_noise_to_random_indices(x, num_indices=0.25, noise_level=[]):
    rand_p = random.uniform(0,1)
    if rand_p >= 0.5:
        sequence_length = x.shape[1]
        indices = random.sample(range(sequence_length), int(sequence_length * num_indices))
            
        # noise = torch.randn(x[:,indices].shape) * args.noise_level
        noise = torch.zeros(x[:,indices].shape)
        
        for i, weight  in enumerate(noise_level):
           noise[i] += torch.randn(x[i, indices].shape) * (weight ** 0.5) * args.noise_level
           
        x[:, indices] += noise
            
    return x

def noise_transform(x):
    return add_noise_to_random_indices(x, num_indices=args.idx_percent, noise_level=var_lst)

