# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 16:00:48 2021

@author: crisprhhx
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
import os
'''

ori = r"L:/湘雅/USER/DATASET/xiangya_selected_10/"
dst = "L:/湘雅/USER/DATASET/xiangya_selected_10_mask/"
for file in os.listdir(ori):
    print(file)
    os.mkdir(dst+file)
    

'''
def std_fy(x, Max=1, Min=0):
    x_std = (x-x.min())/(x.max() - x.min())
    x_scaled = x_std * (Max - Min) + Min
    return x_scaled

a = np.loadtxt(r"DATASET\xiangya_selected_10_mask\CaiZhiyong\IMG-0001-00010_mask.txt")

#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)#默认为范围0~1，拷贝操作

b = std_fy(a,255,0)
np.savetxt("normalized.txt",b)

plt.imshow(b)



