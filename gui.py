from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
import cv2 as cv
import numpy as np
import pandas as pd
import pickle
import joblib
from tkinter import messagebox

a = Tk()
a.title("Diabetics Detector")
a.geometry("1050x570")
a.minsize(1050,570)
a.maxsize(1050,570)

load_model=joblib.load("Project_Saved_Models/d_Trained_RF_model.sav")
#loading standardscaler
scaler=pickle.load(open('Project_Extra/d_scaler_rf.pkl','rb'))

input_labels = ['Age', 'Hypertension', 'Heart Disease', 'Smoking History', 'BMI', 'HbA1c Level', 'Blood Glucose Level']

def prediction():
    values = []
    for entry in input_entries:
        value = entry.get()
        if value == '':
            messagebox.showinfo("Alert", "Fill all fields")
            return
        values.append(float(value))

    list_box.insert(1, "Loading Values")
    
    list_box.insert(3, "Preprocessing")
    list_box.insert(4, values)
    


    column_names = ['age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']

    # Create a DataFrame from the lists
    input_data = pd.DataFrame([values], columns=column_names)

    # Perform scaling
    scaled_input = scaler.transform(input_data)
    list_box.insert(5, "Feature Scaling")
    list_box.insert(6, scaled_input)
    list_box.insert(7, "Loading Trained RF model")
    list_box.insert(8, "\n")
    list_box.insert(9, "Prediction")
    list_box.insert(10, "\n")

    # Perform predictions
    predictions = load_model.predict(scaled_input)

    print(predictions)
    # Get the predicted class label
    predicted_class = predictions[0]

    # Mapping predicted class to attack type
    attack_types = {
        0: 'Healthy',
        1: 'Diabetic',
    }

    predicted_ = attack_types[predicted_class]

    print("Predicted Type:", predicted_)
    list_box.insert(11, predicted_)
    list_box.insert(12, "###########################")

    messagebox.showinfo("Prediction",predicted_)

def Check():


    global f,input_entries
    input_entries = []
    f.pack_forget()

    f = Frame(a, bg="white")
    f.pack(side="top", fill="both", expand=True)

    global f1
    f1 = Frame(f, bg="#f5f0e1")
    f1.place(x=0, y=0, width=700, height=690)
    f1.config()

    input_label = Label(f1, text="INPUT", font="arial 16 bold", bg="#f5f0e1")
    input_label.pack(padx=0, pady=10)

    for i, label in enumerate(input_labels):
        label_widget = Label(f1, text=label + ':', font="arial 16 bold", bg="#f5f0e1")
        label_widget.place(x=50, y=50 + i * 50)
        entry = Entry(f1, font=('Arial', 18), bd=2, width=15)
        entry.place(x=300, y=50 + i * 50)
        input_entries.append(entry)

    predict_button = Button(f1, text="Predict", width=20, height=2, command=prediction, bg="hot pink")
    predict_button.place(x=250, y=450)
  
    f3 = Frame(f, bg="#ff6e40")
    f3.place(x=700, y=0, width=370, height=720)
    f3.config()

    name_label = Label(f3, text="PROCESS", font="arial 16 bold", bg="#ff6e40")
    name_label.pack(pady=20)

    global list_box
    list_box = Listbox(f3, height=12, width=31)
    list_box.pack()


def Home():
    global f
    f.pack_forget()

    f = Frame(a, bg="salmon")
    f.pack(side="top", fill="both", expand=True)

    front_image = Image.open("Experiment/home1.jpg")
    front_photo = ImageTk.PhotoImage(front_image.resize((a.winfo_width(), a.winfo_height()), Image.ANTIALIAS))
    front_label = Label(f, image=front_photo)
    front_label.image = front_photo
    front_label.pack()

    heading_label = Label(f, text="Diabetics Detector", font="arial 26 bold", bg="white")
    heading_label.place(x=550, y=280)

  

f = Frame(a, bg="salmon")
f.pack(side="top", fill="both", expand=True)
front_image1 = Image.open("Experiment/home1.jpg")
front_photo1 = ImageTk.PhotoImage(front_image1.resize((1050,570), Image.ANTIALIAS))
front_label1 = Label(f, image=front_photo1)
front_label1.image = front_photo1
front_label1.pack()

heading_label = Label(f, text="Diabetics Detector", font="arial 26 bold", bg="white")
heading_label.place(x=550, y=280)

m = Menu(a)
m.add_command(label="Home", command=Home)
checkmenu = Menu(m)
m.add_command(label="Check Diabetics", command=Check)
a.config(menu=m)


a.mainloop()
