import numpy as np
import typing
from typing import List
    
def pad(num: int, reversed: bool=False) -> str:
    num_str = str(num)
    while len(num_str) < 3:
        num_str = '0' + num_str
        
    if reversed:
        if len(num_str) < 4:
            num_str = '0' + num_str
            num_str = num_str[::-1]   

    return num_str

def generate_sample(cfg) -> list:
    max = cfg.max_output

    d1 = np.random.randint(cfg.min_output, cfg.max_output)
    d2 = np.random.randint(cfg.min_output, cfg.max_output - d1)

    sum = d1 + d2

    return f'{pad(d1)}{pad(d2)}={pad(sum, reversed=True)}'