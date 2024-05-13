#import essential libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import operator
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot
import pickle
from mlxtend.plotting import plot_confusion_matrix
from select_feature import *

#read dataset using pandas
data = pd.read_csv("Project_Dataset/diabetes_prediction_dataset.csv")
#get first 5 rows
print(data.head())
#view dataframe shape
print(data.shape)

print(data['gender'].value_counts())
print(data['smoking_history'].value_counts())

def convert(data):
  data['gender']=data['gender'].replace("Male",1)
  data['gender']=data['gender'].replace("Female",2)
  data['gender']=data['gender'].replace("Other",3)

  data['smoking_history']=data['smoking_history'].replace("No Info",1)
  data['smoking_history']=data['smoking_history'].replace("never",2)
  data['smoking_history']=data['smoking_history'].replace("former",3)
  data['smoking_history']=data['smoking_history'].replace("current",4)
  data['smoking_history']=data['smoking_history'].replace("not current",5)
  data['smoking_history']=data['smoking_history'].replace("ever",6)

  return data

data=convert(data)

#checking value counts
print(data['diabetes'].value_counts())

#plotiing
data['diabetes'].value_counts().plot(kind='barh')
plt.show()

data['diabetes'].value_counts().plot(kind='pie')
plt.show()

#checking the dataframe contains any null values
print(data.isna().sum())

#perform data division
y=data['diabetes']
x=data.drop(['diabetes'],axis=1)


#Perform feature selection(function call)
fs = select_features(x, y)


column_names=[]
# iterating the columns
for col in x.columns:

    column_names.append(col)


print(column_names)
print(len(column_names))

feature_list=[]
# what are scores for the features
for i in range(len(fs.scores_)):
  # print('Feature %d: %f' % (i, fs.scores_[i]))
  feature_list.append(fs.scores_[i])

print(feature_list)
print(len(feature_list))

#convert to dictionary for better visualization
dictionary = dict(zip(column_names, feature_list))
clean_dict = {k: dictionary[k] for k in dictionary if not pd.isna(dictionary[k])}
# print(clean_dict)
sorted_d = dict( sorted(clean_dict.items(), key=operator.itemgetter(1),reverse=True))

print(sorted_d)
print(len(sorted_d))


#plotting feature scores
names = list(sorted_d.keys())
values = list(sorted_d.values())



plt.figure(figsize=(8,6))
plt.bar(names,values,color='green')
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.1, bottom=0.28, right=0.96, top=0.96)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Feature Score')
plt.show()


#Creating new dataframe based on feature scores
my_data=data[['age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level','diabetes']]
print(my_data.head())
print(my_data.columns)


#########my_data.to_csv("Project_Dataset/preprocessed.csv",index=False)