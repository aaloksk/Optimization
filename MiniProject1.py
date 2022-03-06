# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 19:21:52 2022

@author: aalok
"""


#Import Libraries
import os
import pandas as pd #Reading the data
import numpy as np #Math func
#from matplotlib import pyplot as plt #For plotting
from scipy.optimize import minimize #For minimization
import numdifftools as nd

'''
#Importing the csv file from github repository
#Providing the raw link of csv file
url = 'https://raw.githubusercontent.com/aaloksk/Optimization/main/MiniProject1-Data.csv?token=GHSAT0AAAAAABR2ANNMZT3ZZYNCPQZDHICQYQ4FNXA'

#Reading the data
br = pd.read_csv(url, index_col=0)

#Looking at the imported data
br.head(5)
'''

#Import Data for drive
#Working directory
os.chdir('D:\\Dropbox\\0000Optimization\\MiniProject1')

#Load data
br = pd.read_csv('MiniProject1-Data.csv')

#Looking at the imported data
br.head()

#Arranging Y variable
#Recoding Safe and Unsafe as 1 and 0
br['BridgeCond'] = br['BridgeCond'].replace(to_replace ="S",    value = 1) #Safe as 1
br['BridgeCond'] = br['BridgeCond'].replace(to_replace ="U",    value = 0) #Unsafe as 0

#Calculating Age of bridge
br['Age1'] = (2021-br['YearBuilt']) #On original condition
br['Age2'] = (2021-br['YEAR_RECON']) #Reconstructed bridge
br['Age'] = br[['Age1','Age2']].min(axis=1) #minimum of above two value

br['Age_sq'] = (br['Age'])*(br['Age']) #Geting age_Squared as separate variable


#Our Logistic Regression Function to be optimized
def LL(c,X,Y):
    SLL = 0 #Initializing variable for sum of log likelihood
    
    for i in range(0,999):
        et = c[0] + c[1]*X[0][i] + c[2]*X[1][i] + c[3]*X[2][i] + c[4]*X[3][i] + c[5]*X[4][i] + c[6]*X[5][i]
        p_y_1 = 1/(1+np.exp((-1)*et))
        LL = (Y[i] * np.log(p_y_1)) + ((1-Y[i]) * np.log(1-p_y_1))
        SLL = SLL + LL
    
    return(SLL*(-1))

#Detail statistical description of each variable
zz = br.describe()


#zz.to_csv('Describe.csv', index=False)

#Y-variable
Y = br['BridgeCond']

#Normalize X- variables
br['ADT'] = (br['ADT'] - np.min(br['ADT']))/(np.max(br['ADT']) - np.min(br['ADT']))
br['DeckArea_SQM'] = (br['DeckArea_SQM'] - np.min(br['DeckArea_SQM']))/(np.max(br['DeckArea_SQM']) - np.min(br['DeckArea_SQM']))
br['SeismicHazard'] = (br['SeismicHazard'] - np.min(br['SeismicHazard']))/(np.max(br['SeismicHazard']) - np.min(br['SeismicHazard']))
br['PPTIN'] = (br['PPTIN'] - np.min(br['PPTIN']))/(np.max(br['PPTIN']) - np.min(br['PPTIN']))
br['Age'] = (br['Age'] - np.min(br['Age']))/(np.max(br['Age']) - np.min(br['Age']))
br['Age_sq'] = (br['Age_sq'] - np.min(br['Age_sq']))/(np.max(br['Age_sq']) - np.min(br['Age_sq']))
#Arranging X-variables
X= [br['ADT'], br['DeckArea_SQM'], br['SeismicHazard'], br['PPTIN'],br['Age'],br['Age_sq']]
#X=np.log(X)




#Optimization by Newtons Method

#Initial guess for the beta coefficients
c = [2.0,-1.9,0.3,-0.9,-0.3,-4.0,1.3]

#Params for Newtons Method
tol = 1e-05
eps = 1000000
idx = 0

#c=[0.58,-1.5,2.5,-1.1,0.01]
while(eps>tol):
    a = nd.Gradient(LL)(c,X,Y) #Jacobian
    b = nd.Hessian(LL)(c,X,Y) #Hessian
    binv = np.linalg.inv(b) #Inverse of Hessian
    eta = np.matmul(a,binv) #Matrix multiplication of A and Binverse
    cnew = c - eta #Calculating new values of coefficients
    eps = np.sum(np.sqrt((cnew-c)**2)) #Calcualting the error
    eps = abs(eps) #Getting absolute value
    idx = idx + 1 #Counting number of iterations
    c = cnew #Updating the value of beta coefficients

#Final value of coefficients from Newton's method
beta_newton = c
beta_newton





#Optimization by Neldear Mead
c = [2.0,-1.9,0.3,-0.9,-0.3,-4.0,1.3]

#c = [0.1,0.1,0.1,0.1,0.1,0.1,0.1]
#c=[1,1,1,1,1,1,1]

#Optimization by Neldermead method
#Sending c as intial guess for the coefficients
#Passing X and Y as arguments into minimize model
obj = minimize(LL, c, method='Nelder-Mead',args=(X,Y),options={"maxiter":5000})
obj

#Initializing the dataframe to store optimized coefficients
df = pd.DataFrame({"Beta0" : [obj.x[0]],
                        "Beta1" : [obj.x[1]],
                        "Beta2" : [obj.x[2]],
                        "Beta3" : [obj.x[3]],
                        "Beta4" : [obj.x[4]],
                        "Beta5" : [obj.x[5]],
                        "Beta6" : [obj.x[6]]})

#list_of_no = [i for i in range(1000)]

#Bootstrap Approach
for i in range(0,5300):
    #Resample the original dataframe as 1000 datapoints sampled with replacement
    brs = br.sample(n=1000, replace=True) 
    
    #Reseting the jumbled index value
    brs=brs.reset_index()
    
    #Y from resampled dataset
    Y1 = brs['BridgeCond'] 
    
    #X from resampled dataset
    X1 = [brs['ADT'], brs['DeckArea_SQM'], brs['SeismicHazard'], 
          brs['PPTIN'],br['Age'],br['Age_sq']]
    
    #Intial guess
    c1 = [2.0,-1.9,0.3,-0.9,-0.3,-4.0,1.3]
    #c1 = np.random.sample(7)
        
    #Performing Nelder-Mead for resampled dataset
    obj = minimize(LL, c1, method='Nelder-Mead',args=(X1,Y1))
    
    #Storing the coefficients into a dataframe
    dfi = pd.DataFrame({"Beta0" : [obj.x[0]],
                            "Beta1" : [obj.x[1]],
                            "Beta2" : [obj.x[2]],
                            "Beta3" : [obj.x[3]],
                            "Beta4" : [obj.x[4]],
                            "Beta5" : [obj.x[5]],
                            "Beta6" : [obj.x[6]]})
    
    #Appending the df to store all optimized coefficients
    df = pd.concat([df, dfi])
    
    #Check for loop 
    print(i, obj.success) 

df=df.reset_index() #Reseting the jumbled index value
df #Print the final dataframe

#df.to_csv('Coeff6.csv', index=False)


#Getting Confidence Intervals

#importing the csv with 5000 coefficients
coeffs = pd.read_csv('All_Coeff.csv')

#Getting 2.5,5,95 and 97.5th percventile for each coefficient
b0 = np.percentile(coeffs['Beta0'], [2.5,5,95,97.5])
b1 = np.percentile(coeffs['Beta1'], [2.5,5,95,97.5])
b2 = np.percentile(coeffs['Beta2'], [2.5,5,95,97.5])
b3 = np.percentile(coeffs['Beta3'], [2.5,5,95,97.5])
b4 = np.percentile(coeffs['Beta4'], [2.5,5,95,97.5])
b5 = np.percentile(coeffs['Beta5'], [2.5,5,95,97.5])
b6 = np.percentile(coeffs['Beta6'], [2.5,5,95,97.5])


#Arranging 95% Confidence Level
data=[[b0[0],b0[3]],[b1[0],b1[3]],[b2[0],b2[3]],[b3[0],b3[3]],[b4[0],b4[3]],[b5[0],b5[3]],[b6[0],b6[3]]]
df_coeff = pd.DataFrame(data, columns=['2.5th Percentile', '97.5th Percentile'])
df_coeff

#Arranging 90% Confidence Level
data1=[[b0[1],b0[2]],[b1[1],b1[2]],[b2[1],b2[2]],[b3[1],b3[2]],[b4[1],b4[2]],[b5[1],b5[2]],[b6[1],b6[2]]]
df_coeff1 = pd.DataFrame(data1, columns=['5th Percentile', '95th Percentile'])
df_coeff1



  