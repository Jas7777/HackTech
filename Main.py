# Import the necessary libraries
from PIL import Image
from numpy import asarray
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
# load the image and convert into
# numpy array
import glob
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report
def test_img():
   counter=0
   cv_img = []
   y = []
   for img in glob.glob(r'C:\Users\Sofy\desktop\test9\non-autistic\*.jpg'):
      y.append(0)
      img1 = Image.open(img)
      # Added code
      img1 = img1.resize((100, 100))
      reshaped_image=np.transpose(img1)
      reshaped_image = np.transpose(img1)
      # reshape image being weights are diffrent
      x1 = np.array(img1)
      x = x1.flatten()
      cv_img.append(x)
      counter=counter+1
    #  if(counter==5000):
          #break
   #print(counter)
   for img in glob.glob(r'C:\Users\Sofy\desktop\test9\autistic\*.jpg'):
      y.append(1)
      img1 = Image.open(img)
      img1 = img1.resize((100, 100))
      reshaped_image=np.transpose(img1)
      reshaped_image = np.transpose(img1)
      x1 = np.array(img1)
      x = x1.flatten()
      cv_img.append(x)
      counter=counter+1
    #  if(counter==10000):
    #      break
   #print(counter)
   X = cv_img
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   reg =LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
   reg.fit(X_train, y_train)
   predication=reg.predict(X_test)
   report = classification_report(y_test, predication)
   #print("Classification Report:\n", report)
   # asarray() class is used to convert
   # PIL images into NumPy arrays
   score=0
   counting=0

   predictions = reg.predict(X_train)
   testing = reg.predict(X_test)
   #print(accuracy_score(y_test, testing))
   #print(accuracy_score(y_train, predictions))
   #print(len(X_train))
   #print(len(X_test))
   #exit()

   #print(predictions)

   # testing to predictions
   # test to train
   counting_1=0
   score_1=0
   #for i in range(len(X_test)):
    #  if(y_test[i]==1):
     #     counting=counting+1
      #if(testing[i]==y_test[i] and y_test[i]==1):
       #   score=score+1
      #if(y_test[i]==0):
       #   counting_1=counting_1+1
      #if(testing[i]==y_test[i] and y_test[i]==0):
       #   score_1=score_1+1
   #print(score/counting)# percent of austic prdicted creclty
   #print(score_1/counting_1)# percent of non-austic predicted creclty
   counter=0
   #for k in range(len(X_train)):#train
     # if(predictions[k]==y_train[k]):
       #   counter=counter+1
   #print(counter/len(X_train))
   counter = 0
   #for k in range(len(X_test)):
    #  if (testing[k] == y_test[k]):  # test
     #     counter = counter + 1
   #print("Accuracy",counter / len(X_test))

   img2 = Image.open(r'C:\Users\Sofy\Desktop\test9\test\img10.jpg')
   img2 = img2.resize((100, 100))
   reshaped_image=np.transpose(img2)
   reshaped_image = np.transpose(img2)
   # reshape image being weights are diffrent
   x1 = np.array(img2)
   data = [x1.flatten()]


   #print("test_!")


   return reg.predict(data)





#front-end
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import streamlit as st
import plotly.express as px
img_file = st.camera_input("Take a picture and then crop leaving only your face")
if img_file:
   img = Image.open(img_file)
   cropped_img = st_cropper(img,)
   st.write("Preview")
   _ = cropped_img.thumbnail((150, 150))
   st.image(cropped_img)
   cropped_img.save(r'C:\Users\Sofy\Desktop\test9\test\img10.jpg')
if st.button('Save and Test'):
   output=test_img()
   if(output[0].item())==0:
      st.write("This person tested negitive for autism")
   if(output[0].item())==1:
      st.write("This person tested positive for autism")
