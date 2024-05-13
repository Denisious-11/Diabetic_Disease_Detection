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
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot
import pickle
from mlxtend.plotting import plot_confusion_matrix
from select_feature import *

#read dataset using pandas
data = pd.read_csv("Project_Dataset/preprocessed.csv")

y_final = data['diabetes']
x_final = data.drop(['diabetes'], axis=1)



# TRAIN - TEST SPLITTING
x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.2)
print("\nTraining set")
print(x_train.shape)
print(y_train.shape)
print("\nTesting set")
print(x_test.shape)
print(y_test.shape)

print(x_train)
print(y_train)



#Data balancing
counter = Counter(y_train)
print("BEFORE Balancing : ", counter)

smt = SMOTE(k_neighbors=1)

x_train_sm, y_train_sm = smt.fit_sample(x_train, y_train)

counter = Counter(y_train_sm)
print("AFTER Balancing : ", counter)


smt_test = SMOTE(k_neighbors=1)
x_test_sm, y_test_sm = smt_test.fit_sample(x_test, y_test)

my_x_test_sm=x_test_sm
my_y_test_sm=y_test_sm


#Perform standardization (feature scaling)
scaler = StandardScaler()
x_train_sm = scaler.fit_transform(x_train_sm)
x_test_sm = scaler.transform(x_test_sm)
pickle.dump(scaler,open('Project_Extra/d_scaler_rf.pkl','wb'))



# intialize randomforest classifier
classifier1 = RandomForestClassifier() 
#Training the model
print("Training Progressing")
classifier1.fit(x_train_sm, y_train_sm)
# performing predictions on the test dataset
y_pred1 = classifier1.predict(x_test_sm)
print("Training Finished")

#create confusin matrix
conf_matrix1 = confusion_matrix(y_true=y_test_sm, y_pred=y_pred1)

#plotting
fig, ax = plot_confusion_matrix(conf_mat=conf_matrix1, figsize=(6, 6), cmap=plt.cm.Greens)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.savefig('Project_Extra/d_confusion_matrix_rf.png')
plt.show()

# calculate accuracy
cr = classification_report(y_test_sm, y_pred1)
print(cr)

# model saving
filename1 = "Project_Saved_Models/d_Trained_RF_model.sav"
joblib.dump(classifier1, filename1)


import numpy as np

# Combine the balanced features and corresponding single-label attack column into a DataFrame
balanced_test_data = pd.DataFrame(my_x_test_sm, columns=x_final.columns)
balanced_test_data['diabetes'] = my_y_test_sm

# Save the DataFrame to a CSV file
###########balanced_test_data.to_csv('test.csv', index=False)
