# To set seed random number in order to reproducable results in keras
from numpy.random import seed
seed(4)
import tensorflow
tensorflow.random.set_seed(1234)
########################################
import pandas as pd
from pandas import *
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler #For feature normalization
scaler = MinMaxScaler()




df1 = pd.read_csv("coors_hela.txt",sep=' ',skiprows=(0),header=(0))
df2 = pd.read_csv("hela-reads-ref.eventalign.txt",sep='\t',skiprows=(0),header=(0))

df2.columns=['contig','position','reference_kmer','read_index','strand','event_index','event_level_mean','event_stdv','event_length','model_kmer','model_mean','model_stdv','standardized_level']

#df2 = pd.read_csv("output_no_signals.txt",sep='\t',skiprows=(0),header=(0),nrows=2000)
print(df2.shape)
print("&&&&&&&&")
print(df1.head())
print("***********************")
print(df2.head())
print("######################")

#get the 2nd column of 
#column_2 = df1.iloc[:, 1]

#print(df1['position'].iloc[0:5])
print(df1.iloc[0:5, 1])
print("@@@@@@@@@@@@@@@")
#print(df2['position'].iloc[0:5])
#print(df2.iloc[0:5, 1])
print(df2.iloc[0:5, 9])
print("######################")
'''
model_kmer_list=list(df2.iloc[:, 9]) #10 for model-kmer that
print("333333333333333333", type(model_kmer_list))
print(model_kmer_list[5])
print(model_kmer_list[5][2])
G_kmer_list=[]
for i in model_kmer_list:
    if type(i)!=str:
        print((i))
    if type(i)==str:
        if i[2]=='G':
            G_kmer_list.append(i)

print("length of G_kmer_list",len(G_kmer_list))
print(G_kmer_list[0:50])

df=df2[df2['model_kmer'].isin(G_kmer_list)]
print(df.shape)
print(df.head())

np.savetxt('filtered_df_G_kmer', df,fmt='%s')
'''

#label the data
#kdf=df2
x=list(set(df1.iloc[:,1]).intersection(set(df2.iloc[:,1])))
print("length of intersection list",len(x))


df_Nm=df2[df2['position'].isin(x)]   ####################error here
listofones = [1] * len(df_Nm.index)
# Using DataFrame.insert() to add a column 
df_Nm.insert(13, "label", listofones, True)
df_non_Nm=df2[~df2['position'].isin(x)]
listofzeros=[0]*len(df_non_Nm.index)
df_non_Nm.insert(13, "label", listofzeros, True)
print(df_Nm.shape)
print(df_Nm.head())
print(df_non_Nm.shape)
print(df_non_Nm.head())
#np.savetxt('m2G_samples.txt', df_m2G,fmt='%s')
#np.savetxt('G_samples.txt', df_G,fmt='%s')
##########prepare datast       
df_non_Nm = df_non_Nm.sample(n=len(df_Nm), replace=False) #try replace=false

# Create DataFrame from positive and negative examples
dataset = df_non_Nm.append(df_Nm, ignore_index=True)
dataset['label'] = dataset['label'].astype('category')
#np.savetxt('dataset.txt', dataset,fmt='%s')
dataset.to_csv('Nm_benchmark_hela_sample.csv')          ##Pandas: write to csv File: https://realpython.com/pandas-read-write-files/
