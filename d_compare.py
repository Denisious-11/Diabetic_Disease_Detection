import numpy as np 
import matplotlib.pyplot as plt  
import pickle

def addlabels(methods,values):
    for i in range(len(methods)):
        plt.text(i,values[i],values[i])


def plot1():

    # creating the dataset 
    data = {'RF Model':96,'CNN Model':92,'SVM Model':88} 
    methods = list(data.keys()) 
    values = list(data.values()) 
    
    fig = plt.figure(figsize = (10, 5)) 
    
    # creating the bar plot 
    plt.bar(methods, values, color ='green', width = 0.4)

    addlabels(methods,values)
    
    plt.xlabel("Methods") 
    plt.ylabel("Accuracy") 
    plt.title("Performance comparison ") 
    plt.savefig("Project_Extra/d_compare.png")
    plt.show() 

plot1()