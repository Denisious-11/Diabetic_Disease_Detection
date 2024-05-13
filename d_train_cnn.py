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
from tensorflow.keras.callbacks import ModelCheckpoint 
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
pickle.dump(scaler,open('Project_Extra/d_scaler_cnn.pkl','wb'))


#Applying dimension expansion
x_train_sm = np.expand_dims(x_train_sm, axis=2)
x_test_sm = np.expand_dims(x_test_sm, axis=2)


#model loading
from model import model_cnn

model=model_cnn(x_train_sm)


#saving the model(checkpoint)
checkpoint=ModelCheckpoint("Project_Saved_Models/d_trained_cnn_model.h5",monitor="val_accuracy",save_best_only=True,verbose=1)#when training deep learning model,checkpoint is "WEIGHT OF THE MODEL"
#Training
history=model.fit(x_train_sm, y_train_sm, batch_size=16, epochs=10, validation_data=(x_test_sm, y_test_sm), callbacks=[checkpoint])



#plot accuracy and loss 
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('Project_Extra/d_cnn_acc_plot.png')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('Project_Extra/d_cnn_loss_plot.png')
plt.show()


# Prediction on test data
y_pred = model.predict(x_test_sm)
y_pred = np.round(y_pred).flatten().astype(int)

# Confusion Matrix
conf_mat = confusion_matrix(y_test_sm, y_pred)

# Plotting Confusion Matrix
plt.figure(figsize=(8, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('Project_Extra/d_cnn_confusion_matrix.png')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test_sm, y_pred))
