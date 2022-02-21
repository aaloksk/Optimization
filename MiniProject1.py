# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 19:21:52 2022

@author: aalok
"""


#Import Libraries
import os
import pandas as pd #Reading the data
import numpy as np #Math func
from matplotlib import pyplot as plt #For plotting
from scipy.optimize import minimize #For minimization


#Working directory
os.chdir('D:\\Dropbox\\0000Optimization\\MiniProject1')

#Load data
bd = pd.read_csv('MiniProject1-Data.csv')
bd.head()



#Recoding Safe and Unsafe as 1 and 0
bd['BridgeCond'] = bd['BridgeCond'].replace(to_replace ="S",    value = 1) #Safe as 1
bd['BridgeCond'] = bd['BridgeCond'].replace(to_replace ="U",    value = 0) #Unsafe as 0


#Prob Good = first eqn