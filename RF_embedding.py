# To set seed random number in order to reproducable results in keras
from numpy.random import seed
seed(4)
import tensorflow
tensorflow.random.set_seed(1234)
########################################
from gensim.models import Word2Vec  #for getting kmer embedding
########################################
import pandas as pd
from pandas import *
import numpy as np
import random
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
classifier =RandomForestClassifier(n_estimators=30, max_depth=10, random_state = random.seed(1234))#random_state=0)

import plot_learning_curves as plc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler #For feature normalization

scaler = MinMaxScaler()

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep]#.astype(np.float64)


dataset = pd.read_csv('Nm_benchmark_hela_sample.csv', index_col=0)  #Pandas: read csv file: https://realpython.com/pandas-read-write-files



dataset['mean_diff'] = (dataset['event_level_mean'] - dataset['model_mean']).astype(int)

#shuffle the test and train datasets
###from sklearn.utils import shuffle
###dataset = shuffle(dataset)
dataset['kmer_match'] = np.where((dataset['reference_kmer'] == dataset['model_kmer']), 1, 0)

print(dataset.head())
    
#print(df.columns.tolist())

#cols=['contig', 'position', 'reference_kmer', 'read_index', 'strand', 'event_index','event_level_mean', 'event_stdv',
#      'event_length', 'model_kmer', 'model_mean', 'model_stdv', 'standardized_level', 'kmer_match', 'label']


#columns=['event_stdv','model_stdv','standardized_level','reference_kmer','kmer_match','label']

columns=['position','event_level_mean','event_stdv','model_mean','model_stdv','kmer_match','mean_diff','reference_kmer','label']

#columns=['position','event_level_mean','event_stdv','model_mean','model_stdv','kmer_match','mean_diff','label']



#dataset['label'] = dataset['label'].astype('category')

# #scale training and testing data
dataset=dataset[columns]
dataset=clean_dataset(dataset)

#columns1=['event_stdv','model_stdv','standardized_level','kmer_match']
columns1=['position','event_level_mean','event_stdv','model_mean','model_stdv','kmer_match','mean_diff']

#Feature importance
#columns1=['position']
#columns1=['event_level_mean']
#columns1=['event_stdv']
#columns1=['model_mean']
#columns1=['model_stdv']
#columns1=['kmer_match']
#columns1=['mean_diff']

X = dataset[columns1]
#################################################################################################################

#needed for kmer_embedding
ref_kmer_list=list(dataset['reference_kmer']) 
print(len(ref_kmer_list))


#get kmer_embeddingusing genism
processed_corpus=list(ref_kmer_list)
print("&&&&&&&&",len(processed_corpus))
processed_corpus=[processed_corpus]
print(processed_corpus)
print("#########",len(processed_corpus[0]))
model = Word2Vec(processed_corpus,vector_size=20,min_count=1,window=3)
#model = Word2Vec(processed_corpus,size=100,min_count=1)#, window=5) #min_count=5 leads to an error
print(model)

#words = list(model.wv)
#print(words)
#test='ATACG' in words
#print(",,,",test)
#print("MMMMMMMM",len(words))
#print(model['TATAA'])
#print(type(model['TATAA']))
#print("000000000")
#df=pd.DataFrame(model['TATAA']) 
#df=df.T  #transpose dataframe to convert df of 1 column to df of 1 row
#df.columns = [''] * len(df.columns) #to remove column name
#print("*****",df.columns.tolist()) #to print column name
#print(df)
#print("%%%%%%%%%%",df.shape)

df3=pd.DataFrame()
# For each kmer in ref_kmer_list, find model(kmer)
for i in range(len(ref_kmer_list)):
    x=ref_kmer_list[i]
    #print(type(x))
    ###print(model[x])
    df4=pd.DataFrame(model.wv[x])
    df4=df4.T
    #df4.columns = [''] * len(df4.columns)
    ###print("000000000")
    ###print(df4)
    # to append df4 at the end of df3 dataframe 
    df3 = pd.concat([df3,df4])
print(df3.head)
print("8888888888",df3.shape)    
#df3.to_csv('kmer_embeddings.csv', index_col=0)
######insert embedding of reference-kmer
#print("000000000",X.head) 
df3.reset_index(drop=True, inplace=True)      #To avoid the error at https://vispud.blogspot.com/2018/12/pandas-concat-valueerror-shape-of.html
X.reset_index(drop=True, inplace=True)
X= pd.concat([X,df3],axis=1)
#X=df3  #TO test the effect of embedding only

#########################################
#########################################
print("#############",X.shape)
print(X.head())
print(type(X))
#scale training data
X= scaler.fit_transform(X)
Y = dataset['label'] 
print(",,,,,,,,",X.shape)
print(",,,,,,,,",Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3) for unblanced dataset


#clf = classifier.fit(X_train,y_train)
clf = classifier.fit(X_train,y_train.ravel())



y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)
y_prob = y_prob[:,1]
# Evaluate the model: Model Accuracy, how often is the classifier correct
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report #for classifier evaluation
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score # for printing AUC
from sklearn.metrics import confusion_matrix



print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
 
print(classification_report(y_test, y_pred))
auc=roc_auc_score(y_test.round(),y_pred)
auc = float("{0:.3f}".format(auc))
print("AUC=",auc)
#true negatives c00, false negatives C10, true positives C11, and false positives C01 
#tn c00, fpC01, fnC10, tpC11 
print('CF=',confusion_matrix(y_test, y_pred))
l=confusion_matrix(y_test, y_pred)#https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
print('TN',l.item((0, 0)))
print('FP',l.item((0, 1)))
print('FN',l.item((1, 0)))
print('TP',l.item((1, 1)))
#print(type(X_train), type(y_train))
print("features=", columns1)


'''
from mlxtend.plotting import plot_learning_curves
import matplotlib.pyplot as plt

ax =plot_learning_curves(X_train, y_train, X_test, y_test, clf,  print_model=False , style='classic')
# Adding axes annotations
plt.xlabel('Training set size(Percentage)',fontsize=18)
plt.ylabel('Performance(Misclassification error)',fontsize=18)
plt.xticks(fontsize=18, weight = 'bold') 
plt.yticks(fontsize=18, weight = 'bold')
L=plt.legend(fontsize=20)
L.get_texts()[0].set_text('Training set')
L.get_texts()[1].set_text('Test set')
plt.gcf().set_size_inches(10, 8)
plt.grid(b=None) #for no grid
plt.savefig('RF_loss_test_split.png',dpi=300)
plt.close()


#plot learning curve: works with all classifier and all features except x(padded signal) as it leads to error with SVM 
#References:https://medium.com/@datalesdatales/why-you-should-be-plotting-learning-curves-in-your-next-machine-learning-project-221bae60c53
import matplotlib.pyplot as plt


plc. plot_learning_curves(classifier, X_train, y_train, X_test, y_test)

# Create plot
#plt.title("Learning Curve")
plt.xlabel("Training Set (Size)"), plt.ylabel("Accuracy")#, plt.legend(loc="best")
plt.xticks(fontsize=18, weight = 'bold') 
plt.yticks(fontsize=18, weight = 'bold')
plt.tight_layout()
#plt.show()
plt.savefig('RF_LC_test_split.png',dpi=300)
plt.savefig('RF_LC_test_split.svg',dpi=300)
plt.close()


#plot ROC curve: https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt1

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)


# Print ROC curve
plt1.plot(fpr,tpr)
#plt1.title("ROC Curve")
# axis labels
plt1.xlabel('False Positive Rate')
plt1.ylabel('True Positive Rate')
plt.xticks(fontsize=12, weight = 'bold') 
plt.yticks(fontsize=12, weight = 'bold')
plt1.legend(loc="best")
#plt.show() 
plt1.savefig('RF_ROC_test_split.png',dpi=300)
plt1.savefig('RF_ROC_test_split.svg',dpi=300)
plt1.close()

#############################################
#old code to plot learning curve: works only with RandomForest
#Reference: https://www.dataquest.io/blog/learning-curves-machine-learning/
##################
'''


